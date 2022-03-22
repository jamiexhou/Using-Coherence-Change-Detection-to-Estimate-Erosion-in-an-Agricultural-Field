import snappy
from snappy import ProductIO
from snappy import GPF
from snappy import HashMap
from snappy import WKTReader
import re

# global variable
inputFiles = []

def preprocessing(inloc, infile, outdir, swath):
    '''
    preprocessing(inloc,infile, outdir) is a function that takes file infile found at location inloc and
        preprocesses it separately and leaves the finished file in directory outdir
    preprocessing: str, str, str ->
    output: file
    steps:
    1. TOPSAR-split
    2. Apply orbit file
    '''

    filepath = "{0}/{1}".format(inloc, infile)
    filenamepattern = r"\d{8}T\d{6}_.{6}_.{6}_.{4}"
    outputname = re.findall(filenamepattern, infile)[0]
    writeFormat = 'BEAM-dimap'
    saveloc = "{0}/{1}_{2}".format(outdir, outputname, swath)

    splitParameters = HashMap()
    splitParameters.put('subswath', swath)
    curfile = ProductIO.readProduct(filepath)

    orbitParameters = HashMap()
    orbitParameters.put('continueOnFail', True)

    # processing
    # topsar split
    curfile = GPF.createProduct('TOPSAR-split', splitParameters, curfile)
    # apply orbit file
    curfile = GPF.createProduct('apply-orbit-file', orbitParameters, curfile)

    ProductIO.writeProduct(curfile, saveloc, writeFormat)

def readFilesin(inloc, lof):
    '''healper function for coherenceProcess'''
    global inputFiles
    filepathformat = "{0}/{1}"

    for filepath in lof:
        try:
            file = ProductIO.readProduct(filepathformat.format(inloc, filepath))
            filename = file.getName()
            inputFiles.append(file)
        except Exception:
            print("Oh no an error occured! {0}".format(Exception))
            print("Error on file {0}".format(filepath))


def coherenceProcess(inloc, lof, outloc, ingeometry):
    '''
    coherenceProcess(inloc, lof, outloc) is a function that pocesses the coherence of
        the files in list of files lof, located in path inloc and will place the
        output in path outloc.
    coherenceProcess: str, str, str ->
    steps:
    1. backgeo-encoding
    2. interferogram formation
    3. topsar debust
    4. subset

    process referenced from: https://git.earthdata.nasa.gov/projects/ASR/repos/asf-daac-script-recipes/browse/snappy_topsar_insar_share.py?at=5e0fa89bb1a3de4640771d75df57e4d897042184&raw
    '''

    global inputFiles

    saveFormat = 'BEAM-dimap'
    savename = "{0}/InterferogramStack".format(outloc)

    parametersBackGeo = HashMap()
    parametersBackGeo.put("Digital Elevation Model", "SRTM 3Sec")
    parametersBackGeo.put("DEM Resampling Method", "BICUBIC_INTERPOLATION")
    parametersBackGeo.put("Resamplign Type", "BISINC_5_POINT_INTERPOLATION")
    parametersBackGeo.put("Mask out areas with no elevation", True)
    parametersBackGeo.put("Output Deramp and Demod Phase", False)

    parametersInterferogram = HashMap()
    parametersInterferogram.put("Subtract flat-earth phase", True)
    parametersInterferogram.put("Degree of \"Flat Earth\" polynomial", 5)
    parametersInterferogram.put("Number of \"Flat Earth\" estimation points", 501)
    parametersInterferogram.put("Orbit of interpolation degree", 3)
    parametersInterferogram.put("Include coherence estimation", True)
    parametersInterferogram.put("Square Pixel", False)
    parametersInterferogram.put("Independent Window Sizes", False)
    parametersInterferogram.put("Coherence Azimuth Window size", 10)
    parametersInterferogram.put("Coherence Range Window size", 10)

    parametersDeburst = HashMap()
    parametersDeburst.put("Polarisations", "VV, VH")

    geom = WKTReader().read(ingeometry)
    parametersSubset = HashMap()
    parametersSubset.put('geoRegion', geom)

    # processing

    readFilesin(inloc, lof)
    print("Array of files created")

    curfile = GPF.createProduct("Back-Geocoding", parametersBackGeo, inputFiles)
    print("Back Geo-coding finished")

    curfile = GPF.createProduct("Interferogram", parametersInterferogram, curfile)
    print("interferogram creation finished")

    curfile = GPF.createProduct("TOPSAR-Deburst", parametersDeburst, curfile)
    print("Deburst finished")

    curfile = GPF.createProduct("Subset", parametersSubset, curfile)
    print("Subset finished")

    # save file
    ProductIO.writeProduct(curfile, savename, saveFormat)





def postProcessing(inloc, file, outloc, expr):
    '''gpt -h
    postProcessing:

    steps:
    1. phase displacement
    2. terrain correction
    '''
    
    infilepath = "{0}/{1}".format(inloc, file)
    saveformat = "BEAM-dimap"
    outputpath = "{0}/{1}_{2}".format(outloc, file[0:len(file)-4], "post")

    parameterspd = HashMap()
    BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand = BandDescriptor()
    targetBand.name = 'Coherence'
    targetBand.type = 'uint8'
    targetBand.expression = expr
    targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = targetBand
    parameterspd.put('targetBands', targetBands)

    parameterstc = HashMap()
    parameterstc.put("SourceBands", "Coherence")
    parameterstc.put("demName", "SRTM 1Sec HGT")


    curfile = ProductIO.readProduct(infilepath)

    curfile = GPF.createProduct("BandMaths", parameterspd, curfile)

    curfile = GPF.createProduct("Terrain-correction", parameterstc, curfile)

    ProductIO.writeProduct(curfile, outputpath, saveformat)



