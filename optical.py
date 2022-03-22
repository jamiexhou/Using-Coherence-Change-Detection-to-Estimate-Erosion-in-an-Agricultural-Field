import snappy
from snappy import ProductIO
from snappy import GPF
from snappy import HashMap
from snappy import WKTReader
# from snappy import ProductUtils
# from snappy import Product
import re

def exportNDVI(inloc, infile, outFolder, ingeometry):
    '''
    exportNDVI(inFolder, outFolder): generates ndvi calculation for infile found in location inloc
        and exports it into outFolder. Subset to geometry ingeometry
    exportNDVI: str, str, str->
        inloc: a path
        infile: the sentinel2 file
        outFolder: the output folder
    output: creates a file
    workflow:
        1. Sen2cor processor to change top of atmosphere to bottom of atmosphere
        2. ndvi
        3. subset
    '''

    searchPattern = "\d{8}T\d{6}[_]N\d{4}_R\d{3}_.{6}"
    newName = re.findall(searchPattermyn, infile)[0]
    saveloc = "{0}/{1}".format(outFolder, newName)
    exportFormat = 'GeoTIFF'
    l1cFilepath = "{0}/{1}/{2}".format(inloc, infile, "MTD_MSIL1C.xml")
    # l1cFilepath = "{0}/{1}".format(inloc, infile)
    l2aFilepath = "{0}/{1}/{2}".format(inloc, infile[0:4] + "MSIL2A" + infile[10:],
                                       "MTD_MSIL2A.xml")

    sen2corParameters = HashMap()
    # sen2corParameters.put('aerosol', 'AUTO')
    sen2corParameters.put('cirrusCorrection', 'TRUE')
    sen2corParameters.put('DEMTerrainCorrection', 'TRUE')
    # sen2corParameters.put('midLat', 'AUTO')
    sen2corParameters.put('ozone', '0')
    sen2corParameters.put('resolution', 'ALL')

    ndviparameters = HashMap()
    ndviparameters.put('nirSourceBand', 'B8')
    ndviparameters.put('redSourceBand', 'B4')

    geom = WKTReader().read(ingeometry)
    subsetparameters = HashMap()
    subsetparameters.put('geoRegion', geom)
    # subsetparameters.put('outputImageScaleInDb', False)

    curfile = ProductIO.readProduct(l1cFilepath)

    # Processing
    # sen2cor processor
    sen2corParameters.put('sourceProduct', curfile)
    try:
        curfile = GPF.createProduct('Sen2Cor280', sen2corParameters, curfile)

    except:
        print('"error"')

    # curfile = ProductIO.readProduct(l2aFilepath)

    # ndvi calculation
    curfile = GPF.createProduct('NdviOp', ndviparameters, curfile)

    # subset
    curfile = GPF.createProduct('Subset', subsetparameters, curfile)

    #export
    ProductIO.writeProduct(curfile, saveloc, exportFormat)



