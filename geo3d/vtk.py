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
# VTK helpers
################################################################################
class vtk():

    @staticmethod
    def vtk2da(filename, varname='None'):
        from vtk import vtkStructuredPointsReader
        from vtk.util import numpy_support as VN
        import numpy as np
        import numpy as np
        import xarray as xr

        reader = vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.Update()

        data = reader.GetOutput()
        dim = data.GetDimensions()
        bnd = data.GetBounds()
        values = VN.vtk_to_numpy(data.GetPointData().GetArray(varname))
        values = values.reshape(dim,order='F')

        da = xr.DataArray(values.transpose([2,1,0]),
                    coords=[np.linspace(bnd[4],bnd[5],dim[2]),
                            np.linspace(bnd[2],bnd[3],dim[1]),
                            np.linspace(bnd[0],bnd[1],dim[0])],
                    dims=['z','y','x'])
        return da

    ### Save to VTK (version 1) files
    # use also instead of da2vtk1_scalar_int
    @staticmethod
    def da2vtk_scalar(da, filename):
        import numpy as np
        import sys
        # VTK compatible datatype name string
        # bit, unsigned_char, char, unsigned_short, short, unsigned_int, int, unsigned_long, long, float, or double
        vtk_types = {np.float32: 'float', np.int32: 'int32'}

        # convert range to % range for float data types
        if da.dtype in [np.float32, np.float64]:
            dtype = np.float32
            vals = 100.*(da.values - np.nanmin(da.values))/(np.nanmax(da.values)-np.nanmin(da.values))
        else:
            dtype = np.int32
            vals = da.values

        header = """# vtk DataFile Version 1.0
vtk output
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS %d %d %d
ASPECT_RATIO %f %f %f
ORIGIN %f %f %f
POINT_DATA %d
SCALARS %s %s
LOOKUP_TABLE default
"""          % (da.x.shape[0], da.y.shape[0], da.z.shape[0],
                (da.x[-1] - da.x[0])/(da.x.shape[0]-1),
                (da.y[-1] - da.y[0])/(da.y.shape[0]-1),
                (da.z[-1] - da.z[0])/(da.z.shape[0]-1),
                da.x[0],
                da.y[0],
                da.z[0],
                da.x.shape[0]*da.y.shape[0]*da.z.shape[0],
                da.name, vtk_types[dtype])

        with open(filename, 'wb') as f:
            if sys.version_info >= (3, 0):
                f.write(bytes(header,'utf-8'))
            else:
                f.write(header)
            vals.astype(dtype).byteswap().tofile(f)

    ### Save vector with components (i,j,k) to VTK (version 4.2) binary files
    # ds2vtk3(ds, 'velocity', fname + '.vtk')
    @staticmethod
    def ds2vtk_vector(ds, name, filename):
        import numpy as np
        import sys

        da = ds.transpose('z','y','x')
        header = """# vtk DataFile Version 4.2
vtk output
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS %d %d %d
SPACING %f %f %f
ORIGIN %f %f %f
POINT_DATA %d
VECTORS %s float
"""          % (da.x.shape[0], da.y.shape[0], da.z.shape[0],
                (da.x[-1] - da.x[0])/(da.x.shape[0]-1),
                (da.y[-1] - da.y[0])/(da.y.shape[0]-1),
                (da.z[-1] - da.z[0])/(da.z.shape[0]-1),
                da.x[0],
                da.y[0],
                da.z[0],
                da.x.shape[0]*da.y.shape[0]*da.z.shape[0],
                name)

        with open(filename, 'wb') as f:
            f.write(bytes(header,'utf-8'))
            arr = np.stack([da.i.values, da.j.values, da.k.values],axis=-1)
            np.array(arr, dtype=np.float32).byteswap().tofile(f)

    #writer.WriteToOutputStringOn()
    #writer.Write()
    #binary_string = writer.GetBinaryOutputString()
    @staticmethod
    def vtkpoints2ds(filename_or_binarystring):
        import xarray as xr
        import numpy as np
        #from vtk import vtkStructuredGridReader
        from vtk import vtkStructuredPointsReader
        from vtk.util import numpy_support as VN

        reader = vtkStructuredPointsReader()
        if type(filename_or_binarystring) == bytes:
            reader.ReadFromInputStringOn()
            reader.SetBinaryInputString(filename_or_binarystring, len(filename_or_binarystring))
        else:
            reader.SetFileName(filename_or_binarystring)
        reader.ReadAllScalarsOn()
        reader.Update()

        data = reader.GetOutput()
        dim = data.GetDimensions()
        bnd = data.GetBounds()

        points = data.GetPointData()

        ds = xr.Dataset()
        for idx in range(points.GetNumberOfArrays()):
            arrayname = points.GetArrayName(idx)
            values = VN.vtk_to_numpy(points.GetArray(arrayname))
            values = values.reshape(dim,order='F')

            da = xr.DataArray(values.transpose([2,1,0]),
                        coords=[np.linspace(bnd[4],bnd[5],dim[2]),
                                np.linspace(bnd[2],bnd[3],dim[1]),
                                np.linspace(bnd[0],bnd[1],dim[0])],
                        dims=['z','y','x'])
            ds[arrayname] = da

        return ds
