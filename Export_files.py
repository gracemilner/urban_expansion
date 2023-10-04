'''
Code taken here: "https://gis.stackexchange.com/questions/37238/writing-numpy-array-to-raster-file"
'''

import numpy, sys
from osgeo import gdal
from osgeo.gdalconst import *

def array(LU,Example,Path_Name):
    gdal.AllRegister()
    # get info of example data
    rows=Example.RasterYSize
    cols=Example.RasterXSize
    # create output image
    driver=Example.GetDriver()
    outDs=driver.Create(Path_Name,cols,rows,1,GDT_Float32)  #changed here from GDT_Int32 to Float to allow exported file to have values with decimals
    outBand = outDs.GetRasterBand(1)
    outData = numpy.zeros((rows,cols), numpy.int16)
    # write the data
    outBand.WriteArray(LU, 0, 0)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-1)

    # georeference the image and set the projection
    outDs.SetGeoTransform(Example.GetGeoTransform())
    outDs.SetProjection(Example.GetProjection())

    del outData