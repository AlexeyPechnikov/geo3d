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
# GEE helpers
################################################################################
class gee():

    # GEE function to mask clouds using the Sentinel-2 QA band.
    @staticmethod
    def GEEmaskS2clouds(image):
        # Get the pixel QA band.
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        cloudMask = qa.bitwiseAnd(cloudBitMask).eq(0)
        cirrusMask = qa.bitwiseAnd(cirrusBitMask).eq(0)

        # Return the masked and scaled data, without the QA bands.
        return image\
            .updateMask(cloudMask)\
            .updateMask(cirrusMask)\
            .divide(10000)\
            .select("B.*")\
            .copyProperties(image, ["system:time_start"])

    # GEE function to mask edges on Sentinel-1 GRD image
    @staticmethod
    def GEEmaskS1edges(image):
        edge = image.lt(-30.0)
        maskedImage = image.mask().And(edge.Not())
        return image.updateMask(maskedImage)

    # works for GEE geographic coordinates only, redefine it for projected coordinates
    @staticmethod
    def image2rect(GEEimage, reorder=False):
        import numpy as np

        coords = GEEimage.getInfo()['properties']['system:footprint']['coordinates'][0]
        lats = np.asarray(coords)[:,1]
        lons = np.asarray(coords)[:,0]
        if not reorder:
            return [lons.min(), lats.min(), lons.max(), lats.max()]
        else:
            return [lons.min(), lons.max(), lats.min(), lats.max()]

    # example: redefine library function for projected coordinates
    #def image2rect(GEEimage, reorder=False):
    #    if not reorder:
    #        return [point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius]
    #    else:
    #        return [point[0]-radius, point[0]+radius, point[1]-radius, point[1]+radius]

    # create worldfile to define image coordinates
    # add 1/2 pixel border offset for compability
    @staticmethod
    def worldfile_tofile(fname, GEEimage, dimensions):
        import os

        area = gee.image2rect(GEEimage)
        print (area)
        name, ext = os.path.splitext(fname)
        # use QGIS worldfile names convention
        jext = ext[1] + ext[-1] + 'w'
        fname = os.path.join(str(os.extsep).join([name,jext]))
        with open(fname, 'w') as outfile:
            xres = (area[2]-area[0])/dimensions[0]
            yres = (area[1]-area[3])/dimensions[1]
            coefficients = [xres, 0, 0, yres, area[0]+xres/2, area[3]+yres/2]
            print('\n'.join(map(str, coefficients)), file=outfile)

    # download GEE URL and save to file
    @staticmethod
    def url_tofile(GEEurl, fname):
        import urllib
        import shutil

        with urllib.request.urlopen(GEEurl) as response, open(fname, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)

    @staticmethod
    def preview_tofile(GEEimage, vis, dimensions, fname=None):
        GEEurl = GEEimage\
            .visualize(**vis)\
            .getThumbURL({'dimensions':dimensions, 'format': 'jpg'})
        #print (GEEurl)
        if fname is not None:
            gee.url_tofile(GEEurl, fname)
            gee.worldfile_tofile(fname, GEEimage, dimensions)
        return {'url': GEEurl, 'width': dimensions[0], 'height': dimensions[1]}

    @staticmethod
    def split_rect(GEEimage, n):
        rect = gee.image2rect(GEEimage)
        lats = np.linspace(rect[1], rect[3], n+1)
        lons = np.linspace(rect[0], rect[2], n+1)
        #print (lats, lons)
        cells = []
        for lt1, lt2 in zip(lats.ravel()[:-1], lats.ravel()[1:]):
            for ll1, ll2 in zip(lons.ravel()[:-1], lons.ravel()[1:]):
                cell = [lt1, ll1, lt2, ll2]
                cells.append(cell)
        return cells

    @staticmethod
    def zipsbands2image(files):
        import xarray as xr
        import rioxarray as rio
        import zipfile

        dss = []
        # merge separate file areas
        for fname in sorted(files):
            #print ('fname', fname)
            zip = zipfile.ZipFile(fname)
            # merge separate file to dataset
            ds = xr.Dataset()
            for bandname in zip.namelist():
                varname = bandname.split('.')[1]
                da = rio.open_rasterio(f'/vsizip/{fname}/{bandname}').squeeze(drop=True)
                ds[varname] = da
                da.close()
            dss.append(ds)
        return xr.merge(dss)

    @staticmethod
    def split_rect(rect, n):
        import numpy as np
        lats = np.linspace(rect[0], rect[2], n+1)
        lons = np.linspace(rect[1], rect[3], n+1)
        #print (lats, lons)
        cells = []
        for lt1, lt2 in zip(lats.ravel()[:-1], lats.ravel()[1:]):
            for ll1, ll2 in zip(lons.ravel()[:-1], lons.ravel()[1:]):
                cell = [lt1, ll1, lt2, ll2]
                cells.append(cell)
        return cells
