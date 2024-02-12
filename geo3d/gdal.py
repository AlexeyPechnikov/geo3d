# ----------------------------------------------------------------------------
# Geo3D
#
# This file is part of the Geo3D project: https://github.com/AlexeyPechnikov/geo3d
#
# Copyright (c) 2024, Alexey Pechnikov
#
# Licensed under the BSD 3-Clause License (see LICENSE for details)
# ----------------------------------------------------------------------------

################################################################################
# GDAL functions
################################################################################
class gdal():

    # make dataset GDAL-compatible
    # use as
    # a_ullr = ' '.join(map(str, ds.extent))
    # gdal_translate -of netCDF -co COMPRESS=DEFLATE -co ZLEVEL=6 -co FORMAT=NC4C -a_srs EPSG:{epsg} -a_ullr {a_ullr}
    @staticmethod
    def ds2gdal(ds, attrs_long_names=None):
        if attrs_long_names is not None:
            # set variable name for colorbar plots
            for varname in ds.data_vars:
                ds[varname].attrs['long_name'] = attrs_long_names[varname]
        # calculate dataset extent
        dx = float(ds['x'].diff('x')[0])
        dy = float(ds['y'].diff('y')[0])
        #print ('dx, dy', dx, dy)
        ulx = float(ds.x.min()) - dx/2
        uly = float(ds.y.min()) + dy/2
        lrx = float(ds.x.max()) + dx/2
        lry = float(ds.y.max()) - dy/2
        # for GDAL tools
        ds.coords['projection_x_coordinate'] = ds.x
        ds.coords['projection_y_coordinate'] = ds.y
        # for command line GDAL argument a_ullr
        return (ds, [ulx, uly, lrx, lry])

    @staticmethod
    def rasterize(image, areas, with_nodata=False):
        import xarray as xr
        from rasterio import features
        # increment class value to use 0 as placeholder later
        if 'class' in areas:
            geoms = [(g,c+1) for g,c in zip(areas['geometry'], areas['class'])]
        else:
            geoms = [(g,1) for g in areas['geometry']]
        # rasterio transform is broken, we need to build it from image extent
        # note: gdal uses pixel borders and xarray uses pixel centers
        if 'latitude' in image:
            band = 'latitude'
        else:
            # suppose the same geometries per bands
            band = list(image.data_vars)[0]
        #res = image[band].attrs['res']
        # be careful with ordering
        res = [float(image[band].x.diff('x')[0]), float(image[band].y.diff('y')[0])]
        xmin = image[band].x.values.min()
        ymax = image[band].y.values.max()
        transform = [res[0], 0, xmin - res[0]/2, 0, -res[1], ymax+res[1]/2]
        # rasterize geometries
        da = xr.zeros_like(image[band]).rename('class').astype(np.uint8)
        da.values = np.flipud(features.rasterize(geoms,
                                  dtype=np.uint8,
                                  out_shape=image[band].shape,
                                  transform=transform)) - 1
        df = da.to_dataframe().reset_index()
        if not with_nodata:
            # remove placeholder zero value
            df = df[df['class']<255]
        # return dataarray with placeholder 255 and dataframe
        return da, df

    # vectorize geometries on dask dataarray
    @staticmethod
    def vectorize(image):
        from rasterio import features
        import geopandas as gpd

        # be careful with ordering
        res = [float(image.x.diff('x').mean()), float(image.y.diff('y').mean())]
        xmin = image.x.values.min()
        ymax = image.y.values.min()
        transform = [res[0], 0, xmin - res[0]/2, 0, res[1], ymax+res[1]/2]
        transform

        geoms = (
                {'properties': {'class': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(features.shapes(image.values, mask=None, transform=transform))
        )
        gdf = gpd.GeoDataFrame.from_features(geoms)

        return gdf
