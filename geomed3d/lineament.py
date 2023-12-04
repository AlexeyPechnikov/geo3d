# ----------------------------------------------------------------------------
# GeoMed3D
#
# This file is part of the GeoMed3D project: https://github.com/mobigroup/geomed3d
#
# Copyright (c) 2023, Alexey Pechnikov
#
# Licensed under the BSD 3-Clause License (see LICENSE for details)
# ----------------------------------------------------------------------------

################################################################################
# 3D Lineaments Processing
################################################################################
class lineament():

    @staticmethod
    def geometry_segmentize(line):
        from shapely.geometry import LineString
        return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))

    @staticmethod
    def geometry_azimut(line):
        import math
        coords = line.coords
        angle = math.atan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0])
        # fix to simplify code below
        if math.pi == angle:
            angle = 0
        return angle
        #return math.degrees(angle)
    #print (azimut(df.geometry[0]))

    @staticmethod
    def circle_segment(r, rad, width, xoff=0.0, yoff=0.0, zoff=0.0):
        import numpy as np
        from shapely.geometry import Polygon
        ##(geom, xoff=0.0, yoff=0.0, zoff=0.0)
        from shapely.affinity import translate

        x1 = r*np.cos(rad-width/2)
        y1 = r*np.sin(rad-width/2)
        x2 = r*np.cos(rad+width/2)
        y2 = r*np.sin(rad+width/2)
        return translate(Polygon([[0,0,0],[x1,y1,0],[x2,y2,0],[0,0,0]]),xoff=xoff, yoff=yoff, zoff=zoff)

    #print (circle_segment(radii[0], theta[0]))
    #Polygon(circle_segment(radii[0], theta[0]))

    # force 3D and Z move
    @staticmethod
    def geometry_zmove(geom, zoff=0.0):
        from shapely.affinity import translate
        from shapely.ops import transform

        return translate(transform(lambda x, y: (x, y, 2), geom), xoff=0, yoff=0, zoff=zoff)

    # that's not required when source NetCDF prepared for GDAL
    # transform geodataframe to geo coordinates using corresponding dask dataset
    @staticmethod
    def geometries2dataset(geoms, da, crs=None):
        from shapely.affinity import affine_transform

        # Shapely's affine transform matrix is        [a, b, d, e, xoff, yoff]
        # x′=ax+by+xoff
        # y′=dx+ey+yoff
        # use transform attribute
        a, b, xoff, d, e, yoff = da.transform
        #normal transform
        #transform = [a, b, d, e, xoff, yoff]
        #print ('transform', transform)
        # mirror y per center
        #transform = [a, b, d, -e, xoff, yoff+e*da.shape[0]]
        #print ('transform', transform)

        # calculate transform for dataframe coordinates
        a = float(da.x.diff(dim='x')[0])
        b = 0
        d = 0
        e = float(da.y.diff(dim='y')[0])

        # x coordinate always left to right
        assert(da.x[0] <= da.x[-1])
        xoff = float(da.x.min()) - a/2

        # y coordinate always up to down: mirror y per center
        yoff = float(da.y.min()) + e/2

        # mirror y per center
        transform = [a, b, d, -e, xoff, yoff]
        #print ('transform', transform)

        # transform and set the coordinate system
        out = geoms\
            .apply(lambda g: affine_transform(g, transform))
        if crs is not None:
            return out.set_crs(crs, allow_override=True)
        return out

    # symmetrize directions
    @staticmethod
    def df_symmetrize_angle(df, sectors=180):
        import numpy as np
        import pandas as pd

        # don't modify the input dataframe
        _df = df.copy()

        _df['angle'] = df.angle.apply(lambda radian: radian-np.pi if radian>=0 else np.pi+radian)
        df_sym = pd.concat([df, _df])

        # Pandas 0.25.0+
        angles = np.linspace(-np.pi, np.pi, sectors+1, endpoint=True)
        labels = (angles[1:] + angles[:-1])/2
        df_sym['sector'] = labels[np.digitize(df_sym.angle, angles)-1]
        return df_sym.groupby(['sector']).agg(num = ('sector','count'),length = ('length','sum')).reset_index()

    @staticmethod
    def df2lineaments(df, sigma, resolution, x0=0, y0=0):
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString

        # don't modify the input dataframe
        df = df.copy()
        # transform to geo coordinates
        #df['geometry'] = geometries2dataset(df['geometry'], raster, gdf.crs)

        # any small enough values, 1/10 pixel id ok
        df['geometry'] = df.geometry.simplify(0.1*resolution, preserve_topology=False)

        # we need 2+ pixels or a half wavelength per lineament
        df['length'] = df.geometry.length
        #minscale = np.max([2.0*resolution, sigma/4])
        minscale = np.max([2.0*resolution, 2*sigma/16])
        #print ('minscale', minscale)
        df = df[df['length']>minscale]

        ## START TODO
        # save contours per wavelength
        contour_df = df[['geometry','length']].copy()
        # calculate 'z' coordinate by wavelength
        contour_df['z'] = -sigma/np.sqrt(2)

        # calculate lineaments density
        contour_df['density'] = contour_df.geometry.apply(lambda g: np.sum(contour_df.intersects(g.buffer(sigma/4), align=False)))
        # sometimes we have just 0 intersections... weird
        contour_df['density'] = contour_df['density'].apply(lambda val: val-1 if val>=1 else 0)
        contour_df['density'] = contour_df['density']/contour_df['density'].max()
        # force 3d
        contour_df['geometry'] = contour_df.apply(lambda row: geometry_zmove(row.geometry, row.z), axis=1)
        # collect to results
        #contour_dfs.append(contour_df)
        ## END TODO

        #print ('contours range', np.round(df.z.min(),2), np.round(df.z.max(),2))
        df['geometry'] = df.geometry.apply(lambda geom: MultiLineString([geom]) if geom.geom_type == 'LineString' else geom)
        df = df.explode("geometry").reset_index(drop=True)
        #print ('geometries1',len(df))

        # split geometry segments to list and split the lists to single geometries
        df['geometries'] = df.geometry.apply(lambda geom: geometry_segmentize(geom))
        df = df.explode("geometries").reset_index(drop=True)
        df = df.rename(columns={'geometry':'_','geometries': 'geometry'}).drop(['_'], axis=1)
        #print ('geometries2',len(df))

        df['angle'] = df.geometry.apply(geometry_azimut)
        df['length'] = df.geometry.length

        # we need 2+ pixels or a half wavelength per lineament
        #df = df[df['length']>=2.0]
        #minscale = np.max([2.0*resolution, sigma/4])
        minscale = np.max([2.0*resolution, sigma/16])
        #print ('minscale', minscale)
        df = df[df['length']>minscale]
        #print ('geometries3',len(df))

        # symmetrize directions
        df_sym_sector = df_symmetrize_angle(df)

        # calculate plot
        theta = df_sym_sector.sector.values
        radii = df_sym_sector.length.values
        #print ('theta', theta[0], 'radii', radii[0])
        #width = np.diff(angles)[0]
        width = np.asarray(df_sym_sector.sector.sort_values().diff())[-1]

        # save plot as geometries
        #orient_df = gpd.GeoDataFrame({'r':radii/np.max(radii), 'theta': theta, 'z': -sigma/np.sqrt(2)},
        #                       geometry=[circle_segment(r, angle, xoff=x0, yoff=y0, zoff=sigmas[-1]-sigma) for (r, angle) in zip(sigma*radii/np.max(radii), theta)])
        orient_df = gpd.GeoDataFrame({'r':radii/np.max(radii), 'theta': theta, 'z': -sigma/np.sqrt(2)},
                               geometry=[circle_segment(r, angle, width, xoff=x0, yoff=y0, zoff=-sigma/np.sqrt(2)) for (r, angle) in zip(sigma*radii/np.max(radii), theta)])
        #colors = cm.jet((orient_df.r-orient_df.r.min())/(orient_df.r.max()-orient_df.r.min()))
        #print (orient_df.r.min(), orient_df.r.max())
        #orient_df.to_file('../data/processed/plot.geojson', driver='GeoJSON')
        #orient_df.plot(color=colors, alpha=0.5)
        #orient_dfs.append(orient_df)

        return (contour_df, orient_df)
