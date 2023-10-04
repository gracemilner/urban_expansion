'''
input is a raster file (.rst, .tif), output a 2D array
'''

from osgeo import gdal

def rst(file):
    Raster = gdal.Open(file)
    Band=Raster.GetRasterBand(1)
    Array=Band.ReadAsArray()
    #Array[Array<0]=0  #(removed code which removed any values below zero)
    return(Array)
