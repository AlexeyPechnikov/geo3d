# ----------------------------------------------------------------------------
# GeoMed3D
#
# This file is part of the GeoMed3D project: https://github.com/mobigroup/geomed3d
#
# Copyright (c) 2023, Alexey Pechnikov
#
# Licensed under the BSD 3-Clause License (see LICENSE for details)
# ----------------------------------------------------------------------------
import numpy as np
import xarray as xr
import pandas as pd
from numba import jit
import numba as nb
import dask

################################################################################
# 3D Inversion Processing
################################################################################
class geomed3d():

    @staticmethod
    def ring(r):
        A = np.arange(0,r+1)**2
        dists = np.sqrt( A[:,None] + A)
        mask = -np.ones(dists.shape)
        for ridx in range(r+1):
            mask[(dists<=ridx) & (dists>ridx-1)] = ridx
        out = np.zeros((2*r+1,2*r+1))
        out[:r+1,r:] = np.flipud(mask)
        out[r:,r:] = mask
        out[r:,:r+1] = np.fliplr(mask)
        out[:r+1,:r+1] = np.fliplr(np.flipud(mask))
        return out.astype(int)

    #unitmask = unit_ring_qpy(r)
    #print (unitmask)
    #plt.imshow(unitmask, interpolation='None')
    #plt.colorbar()

    @staticmethod
    @jit(nopython=True, parallel=False)
    def focal_stat(yx, r, array, matrix):
        # define empty result when processing impossible
        nodata = (np.nan*np.empty((2,r))).astype(np.float32)
        #print ('geomed_stat', yx, r, array.shape, matrix.shape)
        y, x = yx
        # return original values to test
        #return (array[y,x]*np.ones((2,r))).astype(np.float32)
        # check window location inside raster
        ysize, xsize = array.shape
        if x < r or y < r or x >= xsize - r or y >= ysize - r:
            return nodata
        window = array[y-r:y+r+1, x-r:x+r+1].ravel()
        #print ('window', window)
        if window.size == 0:
            return nodata
        # ignore NaN values
        mask = ~np.isnan(window)
        # initialize empty statistics
        means1 = np.zeros(r, dtype=np.float64)
        means2 = np.zeros(r, dtype=np.float64)
        # counters for pixels
        counts = np.zeros(r, dtype=np.int32)
        for m, v in zip(matrix[mask], window[mask]):
            if m <= 0:
                continue
            means1[m-1] += v
            means2[m-1] += v**2
            counts[m-1] += 1
        # normalize stats
        means1 = 1.*means1/counts
        means2 = 1.*means2/counts
        # return statistics calculated for 4+ pixels only
        outmean = np.where(counts>=4, means1, np.nan)
        outstd  = np.where(counts>=4, means2 - means1**2, np.nan)
        # return stacked statistics
        return np.stack((outmean, np.sqrt(outstd)/outmean)).astype(np.float32)

    #out = geomed_stat(np.array([1,1]), r, image.values, unitmask.ravel())
    #print (out.shape)
    #print (out)
    @staticmethod
    def compute(da, df_mask, r, chunksize=256):
        # check image resolution
        dy = float(da.y.diff(dim='y').mean())
        dx = float(da.x.diff(dim='x').mean())
        assert abs(np.round(dy)) == abs(np.round(dx)), f'Image y={dy}, x={dx} resolutions should be the same magnitude'
        # add raster pixel coordinates
        da = da.copy()
        da.coords['iy'] = xr.DataArray(np.arange(da.y.size), dims=['y'])
        da.coords['ix'] = xr.DataArray(np.arange(da.x.size), dims=['x'])
        # find array pixels for the mask
        mask = da.sel(y=xr.DataArray(df_mask.y), x=xr.DataArray(df_mask.x), method='nearest')

        # prepare circle mask
        unitmask = geomed3d.ring(r)
        pixels = mask.rename('mask').to_dataframe()[['iy','ix']].values
        coords = df_mask[['y','x']].values

        # define wrapper function
        calculate = lambda yxs: np.apply_along_axis(geomed3d.focal_stat, 1, yxs, r, da.values, unitmask.ravel())

        stats = xr.apply_ufunc(
            calculate,
            xr.DataArray(pixels).chunk(chunksize),
            vectorize=False,
            dask='parallelized',
            input_core_dims=[['dim_1']],
            output_core_dims=[('stats','z')],
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'output_sizes': {'stats': 2, 'z': r}}
        ).load()

        # vertical shift for the cube, pixels
        dzs = np.round(df_mask.z/(dx/np.sqrt(2))).astype(int)
        # shift up cube's top level, pixels
        dzmax = np.max(dzs)

        # shift data cube vertical columns, pixels
        data = stats.values
        for idx, (values, dz) in enumerate(zip(data, (dzmax - dzs).values)):
            if dz > 0:
                data[idx] = np.concatenate((np.nan*np.empty((2, dz if dz<r else r)), values[:,:-dz]), axis=1)

        # convert sparse statistics array to datatarray
        ys = coords[:,0]
        xs = coords[:,1]
        das = [pd.DataFrame({
                            'y': ys,
                            'x': xs,
                            'density':     stat[0],
                            'anomaly':     stat[1]
                            }).set_index(['y', 'x']).to_xarray()
               for stat in stats.T
              ]
        ds = xr.concat(das, dim='z')

        # define depths by source image resolution and scale factor
        ds['z'] = ((dzmax -(ds['z'] + 1))*dx/np.sqrt(2))
        return ds

    # stats = geomed(image, df_mask, r=100, nodask=True)
    # stats = geomed(image, df_mask, r=100)

    @staticmethod
    def gaussian_range(raster0, g1, g2, backward=False):
        import numpy as np
        from scipy.ndimage.filters import gaussian_filter

        raster = raster0.copy()
        raster.values = raster.values.astype(np.float32)
        if backward:
            raster.values = gaussian_filter(raster.values, g1) \
                - gaussian_filter(raster.values,g2)
        else:
            raster.values = gaussian_filter(raster.values,g1,mode='constant', cval=np.nan) \
                - gaussian_filter(raster.values, g2, mode='constant', cval=np.nan)
        return raster

    @staticmethod
    def gaussian(raster0, g, backward=False):
        import numpy as np
        from scipy.ndimage.filters import gaussian_filter

        raster = raster0.copy()
        if backward:
            raster.values = gaussian_filter(raster.values.astype(np.float32),g)
        else:
            raster.values = gaussian_filter(raster.values.astype(np.float32), g, mode='constant', cval=np.nan)
        return raster

    @staticmethod
    def spectrum(raster, sigmas, scale):
        import xarray as xr

        rasters = []
        print (f'Calculate spectrum: {len(sigmas)} wavelengths')
        for g in sigmas:
            print (".", end = '')
            _raster = geomed3d.gaussian_range(raster, g-.5, g+.5, backward=True)
            _raster['r'] = g*scale
            rasters.append(_raster)
        print ()
        return xr.concat(rasters, dim='r')

    @staticmethod
    def correlogram(raster1, raster2, sigmas, scale):
        import xarray as xr
        import numpy as np
        import pandas as pd

        spectrum1 = geomed3d.spectrum(raster1, sigmas, scale)
        spectrum2 = geomed3d.spectrum(raster2, sigmas, scale)

        corrs = []
        print (f'Calculate correlogram: {len(sigmas)} wavelengths')
        for ridx in range(len(sigmas)):
            print (".", end = '')
            _spectrum2 = spectrum2[ridx]
            for didx in range(len(sigmas)):
                _spectrum1 = spectrum1[didx]
                df = pd.DataFrame({'r1': _spectrum1.values.flatten(), 'r2': _spectrum2.values.flatten()})
                corr = round((df.corr()).iloc[0,1],2)
                corrs.append(corr)
        print ()
        da_corr = xr.DataArray(np.array(corrs).reshape([len(sigmas),len(sigmas)]),
                              coords=[scale*sigmas,scale*sigmas],
                              dims=['r2','r1'])

        return da_corr

    @staticmethod
    def crop_histogram(da, q=[5, 95]):
        if q is None:
            return da
        pcnt = np.nanpercentile(da.values.reshape(-1), q)
        return xr.DataArray(np.clip(da.values, pcnt[0], pcnt[1]), coords=da.coords).rename(da.name)
