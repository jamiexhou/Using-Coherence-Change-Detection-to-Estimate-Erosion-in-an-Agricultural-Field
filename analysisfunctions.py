from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle


# np.set_printoptions(threshold=np.inf)
nodata = -9999

def Union(lst1, lst2):
    '''
    Union(lst1, lst2) is a helper function that gets the union of two lists
    :param lst1: listof(any)
    :param lst2: listof(any)
    :return: listof(any)
    Examples:
        Union([1,2,3], [4,5,6]) -> [1,2,3,4,5,6]
        Union([1,2,3], [3,4,5]) -> [1,2,3,4,5]
    '''
    final_list = sorted(lst1 + lst2)
    return final_list

def getCorrelation(ry, rx, yband):
    '''
    getCorrelation(ry, rx, saveloc, savename) is a function that prints the correlation between the two
        raster datasets ry and rx.
    :param ry: raster dataset. Must be same dimensions as raster rx
    :param rx: raster dataset. Must be same dimensions as raster ry
    :return: None
    '''

    # get the data and put in a 1d format
    xraster = gdal.Open(rx)
    print("The x of this band is {0}, y is {1}".format(xraster.RasterXSize, xraster.RasterYSize))
    xband = xraster.GetRasterBand(1)
    xdata = (xband.ReadAsArray()).flatten()
    yraster = gdal.Open(ry)
    print("The x of this band is {0}, y is {1}".format(yraster.RasterXSize, yraster.RasterYSize))
    yband = yraster.GetRasterBand(yband)
    ydata = yband.ReadAsArray().flatten()

    # get rid of what we dont need
    del xraster, xband, yraster, yband

    # print("dim of xraster is {0}".format(xraster.shape))
    print("xdata is length {0}".format(len(xdata)))
    # print("dim of yraster is {0}".format(yraster.shape))
    print("ydata is length {0}".format(len(ydata)))

    # get rid of nodata values.
    # get all nodata values
    xremove_list = []
    yremove_list = []
    for i in range(0, len(xdata)):
        if xdata[i] == nodata:
            xremove_list.append(i)
        if (ydata[i] == nodata): # or ydata[i] == 0):
            yremove_list.append(i)

    removelist = Union(xremove_list, yremove_list)
    # print("there is no data in {0}".format(removelist))

    xdata_clean = xdata
    ydata_clean = ydata
    # remove all nodata values
    for i in removelist[::-1]:
        xdata_clean = np.delete(xdata_clean, i)
        ydata_clean = np.delete(ydata_clean, i)

    del xdata, ydata

    print("xdata is length {0}".format(len(xdata_clean)))
    print("ydata is length {0}".format(len(ydata_clean)))

    print("correlation is {0}".format(np.corrcoef(xdata_clean, ydata_clean)))
    print("scipy says correlation is {0}".format(stats.pearsonr(xdata_clean, ydata_clean)))
    # make figure
    fig, ax = plt.subplots(figsize = (15,8))

    # plotting data
    plt.scatter(xdata_clean, ydata_clean, s=0.1, c = 'black')

    # making the graph pretty and informative
    plt.title("Raster Data Scatter Plot", fontsize = 28)
    plt.xlabel("In-situ comparison", fontsize = 22)
    plt.ylabel("combined coherence value", fontsize = 22)
    plt.show()



def getRegression(ry, rx, listofndvi, yband):
    '''
    getRegression(ry, rx, listofndvi, saveloc, savename): Gets the linear regression between ry,
        rx and the list of ndvi
    :param ry: a raster dataset with same dimensions as rx and listofndvi
    :param rx: a raster dataset with same dimensions as ry and listofndvi
    :param listofndvi: listof(Raster), must have same dimensions as rx and ry
    :return: none
    '''
    # 2 lambda functions used to transform the shape of the input into regression
    innerList = lambda ll, n: [ll[i][n] for i in range(0, len(ll))]
    transpose = lambda ll: [innerList(ll, i) for i in range(0, len(ll[0]))]


    # get the data and put in a 1d format
    xraster = gdal.Open(rx)
    xband = xraster.GetRasterBand(1)
    print("The x of this band is {0}, y is {1}".format(xraster.RasterXSize, xraster.RasterYSize))
    # print("The shape is {0}".format(xband.shape))
    xdata = (xband.ReadAsArray()).flatten()
    yraster = gdal.Open(ry)
    yband = yraster.GetRasterBand(yband)
    ydata = yband.ReadAsArray().flatten()

    # get rid of what we dont need
    del xraster, xband, yraster, yband

    # print("dim of xraster is {0}".format(xraster.shape))
    print("xdata is length {0}".format(len(xdata)))
    # print("dim of yraster is {0}".format(yraster.shape))
    print("ydata is length {0}".format(len(ydata)))

    # get median of ndvi bands
    ndvidatalist = []
    # get all the ndvi bands first
    for tile in listofndvi:
        ndviraster = gdal.Open(tile)
        ndviband = ndviraster.GetRasterBand(1)
        ndvidata = ndviband.ReadAsArray().flatten()
        print("The x of this band is {0}, y is {1}".format(ndviraster.RasterXSize, ndviraster.RasterYSize))
        ndvidatalist.append(ndvidata)
        print("AAA dirs {0}".format(len(ndvidata)))

    # get rid of nodata values.
    # get all nodata values
    xremove_list = []
    yremove_list = []
    for i in range(0, len(xdata)):
        if xdata[i] == nodata:
            xremove_list.append(i)
        if ydata[i] == nodata:
            yremove_list.append(i)

    removelist = Union(xremove_list, yremove_list)

    xdata_clean = xdata
    ydata_clean = ydata
    ndvidatalist_clean = ndvidatalist
    # remove all nodata values
    for i in removelist[::-1]:
        xdata_clean = np.delete(xdata_clean, i)
        ydata_clean = np.delete(ydata_clean, i)

        for j in range(0, len(ndvidatalist_clean)):
            ndvidatalist_clean[j] = np.delete(ndvidatalist_clean[j], i)

    del xdata, ydata, ndvidatalist

    del listofndvi

    # print everything out just to be sure
    print("xdata is length {0}".format(len(xdata_clean)))
    print("ydata is length {0}".format(len(ydata_clean)))

    print("below is the ndvi lists")
    for i in range(0, len(ndvidatalist_clean)):
        print("ndvi {0} is length {1}".format(i, len(ndvidatalist_clean[i])))

    explanatory = ndvidatalist_clean
    explanatory.append(xdata_clean)
    explanatory=transpose(explanatory)

    reg = LinearRegression().fit(explanatory, ydata_clean)
    print("Regression is {0}".format(reg.score(explanatory, ydata_clean)))


def getForest(ry, rx, listofndvi, yband):
    '''
    getForest(ry, rx, listofndvi, saveloc, savename): uses a random forest regression to predict ry as a
        function of rx and listofndvi. Use training and validation datasets of 0 to 800 and 801 to 1173 respectively
        Used this tutorial: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py
    :param ry: a raster dataset with same dimensions as rx and listofndvi
    :param rx: a raster dataset with same dimensions as ry and listofndvi
    :param listofndvi: listof(Raster), must have same dimensions as rx and ry
    :return: None
    '''
    # 2 lambda functions used to transform the shape of the input into regression
    innerList = lambda ll, n: [ll[i][n] for i in range(0, len(ll))]
    transpose = lambda ll: [innerList(ll, i) for i in range(0, len(ll[0]))]

    # get the data and put in a 1d format
    xraster = gdal.Open(rx)
    xband = xraster.GetRasterBand(1)
    print("The x of this band is {0}, y is {1}".format(xraster.RasterXSize, xraster.RasterYSize))
    xdata = (xband.ReadAsArray()).flatten()
    yraster = gdal.Open(ry)
    yband = yraster.GetRasterBand(yband)
    print(yband)
    ydata = yband.ReadAsArray().flatten()

    # get rid of what we dont need
    del xraster, xband, yraster, yband

    # print("dim of xraster is {0}".format(xraster.shape))
    print("xdata is length {0}".format(len(xdata)))
    # print("dim of yraster is {0}".format(yraster.shape))
    print("ydata is length {0}".format(len(ydata)))

    # get median of ndvi bands
    ndvidatalist = []
    # get all the ndvi bands first
    for tile in listofndvi:
        ndviraster = gdal.Open(tile)
        ndviband = ndviraster.GetRasterBand(1)
        ndvidata = ndviband.ReadAsArray().flatten()
        print("The x of this band is {0}, y is {1}".format(ndviraster.RasterXSize, ndviraster.RasterYSize))
        ndvidatalist.append(ndvidata)
        print("AAA dirs {0}".format(len(ndvidata)))

    # get all nodata occurrences
    removelistndvi = []
    for band in ndvidatalist:
        removelistndvitemp = []
        for i in range(0, len(band)):
            if band[i] == nodata:
                removelistndvi.append(removelistndvitemp)

    # get rid of nodata values.
    # get all nodata values
    xremove_list = []
    yremove_list = []
    for i in range(0, len(xdata)):
        if xdata[i] == nodata:
            xremove_list.append(i)
        if ydata[i] == nodata:
            yremove_list.append(i)

    removelist = Union(xremove_list, yremove_list)
    for smalllist in removelistndvi:
        removelist = Union(removelist, smalllist)
    # print("there is no data in {0}".format(removelist))

    # print(ndvidatalist)

    xdata_clean = xdata
    ydata_clean = ydata
    ndvidatalist_clean = ndvidatalist
    # remove all nodata values
    for i in removelist[::-1]:
        xdata_clean = np.delete(xdata_clean, i)
        ydata_clean = np.delete(ydata_clean, i)

        for j in range(0, len(ndvidatalist_clean)):
            ndvidatalist_clean[j] = np.delete(ndvidatalist_clean[j], i)

    del xdata, ydata, ndvidatalist

    del listofndvi

    # print everything out just to be sure
    print("xdata is length {0}".format(len(xdata_clean)))
    print("ydata is length {0}".format(len(ydata_clean)))

    print("below is the ndvi lists")
    for i in range(0, len(ndvidatalist_clean)):
        print("ndvi {0} is length {1}".format(i, len(ndvidatalist_clean[i])))

    explanatory = ndvidatalist_clean
    explanatory.append(xdata_clean)
    explanatory = transpose(explanatory)
    # print(explanatory)

    reg1 = RandomForestRegressor(max_depth = 2, random_state = 0)
    reg1.fit(explanatory, ydata_clean)
    # suffle datasets
    explanatory, ydata_clean = shuffle(explanatory, ydata_clean, random_state = 0)

    training = explanatory[:int(np.round(7*(len(explanatory))/10))]
    validation = explanatory[int(np.round(7*(len(explanatory))/10)) + 1:]
    predtrain = reg1.predict(training)
    predval = reg1.predict(validation)

    plt.figure()
    plt.plot(predtrain, "b^", label = "training")
    plt.plot(predval, "gd", label = "validation")

    plt.show()

    print("Correlation in the training dataset is {0}".format(stats.pearsonr(predtrain, ydata_clean[:int(np.round(7*(len(explanatory))/10))])))
    print("Correlation in the validation dataset is {0}".format(stats.pearsonr(predval, ydata_clean[int(np.round(7*(len(explanatory))/10))+ 1:])))

def predictErosion(ry, rx_vv, rx_vh, listofndvi, ybandnum, ybenchmark, predpercentile):
    '''
    predictErosion(ry, rx_vv, rx_vh, listofndvi, ybandnum, ybenchmark, predpercentile): uses random forest modeling to
        predict erosion, using the vv coherence band for rx_vv and the vh coherence bands for rx_vh. The pixels with
        values above the percentile at predpercentile pixels are considered 'significant', to be compared with the
        erosion on the ry band, using ybenchmark.
    :param ry: the in-situ results
    :param rx_vv: coherence product VV band
    :param rx_vh: coherence product VH band
    :param listofndvi: list of ndvi over the time period
    :param ybandnum: the product i was using for ry had multiple bands. For ease of use, this species each band
    :param ybenchmark: benchmark for what is considered erosion on the ry data
    :param predpercentile: anything above this percentile value is considered eroded
    :return:
    '''

    # 2 lambda functions used to transform the shape of the input into regression
    innerList = lambda ll, n: [ll[i][n] for i in range(0, len(ll))]
    transpose = lambda ll: [innerList(ll, i) for i in range(0, len(ll[0]))]

    print("Y benchmark of {0} and ypredicted benchmark of {1}".format(ybenchmark, predpercentile))

    # get resultant data

    yraster = gdal.Open(ry)
    yband = yraster.GetRasterBand(ybandnum)
    ydata = yband.ReadAsArray().flatten()

    # get vv vh data
    vvraster = gdal.Open(rx_vv)
    vvband = vvraster.GetRasterBand(1)
    vvdata = (vvband.ReadAsArray()).flatten()

    vhraster = gdal.Open(rx_vh)
    vhband = vhraster.GetRasterBand(1)
    vhdata = (vhband.ReadAsArray()).flatten()

    ndvidatalist = []
    # get ndvi data
    for tile in listofndvi:
        ndviraster = gdal.Open(tile)
        ndviband = ndviraster.GetRasterBand(1)
        ndvidata = ndviband.ReadAsArray().flatten()
        print("The x of this band is {0}, y is {1}".format(ndviraster.RasterXSize, ndviraster.RasterYSize))
        ndvidatalist.append(ndvidata)
        print("AAA dirs {0}".format(len(ndvidata)))

    # get rid of nodata values.
    # get all nodata values
    removelist = []
    for i in range(0, len(vvdata)):
        if (vvdata[i] == nodata or vhdata[i] == nodata or ydata[i] == nodata):
            removelist.append(i)

    print("{0} values have had to be removed".format(len(removelist)))

    vvdata_clean = vvdata
    vhdata_clean = vhdata
    ydata_clean = ydata
    ndvidatalist_clean = ndvidatalist
    # remove all nodata values
    for i in removelist[::-1]:
        vvdata_clean = np.delete(vvdata_clean, i)
        vhdata_clean = np.delete(vhdata_clean, i)
        ydata_clean = np.delete(ydata_clean, i)

        for j in range(0, len(ndvidatalist_clean)):
            ndvidatalist_clean[j] = np.delete(ndvidatalist_clean[j], i)

    del vvdata, vhdata, ydata, ndvidatalist

    print("New list lengths are {0} | {1} | {2} for vv, vh and y respectively".format(len(vvdata_clean),
                                                                                      len(vhdata_clean),
                                                                                      len(ydata_clean)))

    # prep the explanatory variables
    explanatory = ndvidatalist_clean
    explanatory.append(vvdata_clean)
    explanatory.append(vhdata_clean)
    print("AAAAAAA {0}".format(np.shape(explanatory)))
    explanatory = transpose(explanatory)

    # print(explanatory)

    reg1 = RandomForestRegressor(max_depth = 2, random_state = 0)
    reg1.fit(explanatory, ydata_clean)
    # randomize the list
    # explanatory, ydata_clean = zip(*sorted(zip(explanatory, ydata_clean)))
    explanatory, ydata_clean = shuffle(explanatory, ydata_clean, random_state = 0)

    print("Partition at {0} for dataset {1}".format(int(np.round(7*(len(explanatory))/10)), np.shape(explanatory)))
    training = explanatory[:int(np.round(7*(len(explanatory))/10))]
    validation = explanatory[int(np.round(7*(len(explanatory))/10)) + 1:]
    predtrain = reg1.predict(training)
    predval = reg1.predict(validation)

    # print(predtrain)

    # double check some of the accuracy
    print("Correlation in the training dataset is {0}".format(stats.pearsonr(predtrain, ydata_clean[:int(np.round(7*(len(explanatory))/10))])))
    print("Correlation in the validation dataset is {0}".format(stats.pearsonr(predval, ydata_clean[int(np.round(7*(len(explanatory))/10))+ 1:])))

    # make a regressor off some of the training data
    predictedAll = reg1.predict(explanatory)

    # reclassify the ydata to 'erosion/no erosion'
    ydata_reclass = ydata_clean
    for i in range(0, len(ydata_reclass)):
        if (ydata_reclass[i] >= ybenchmark):
            ydata_reclass[i] = 1
        else:
            ydata_reclass[i] = 0

    # reclassify the predicted to 'erosion/no erosion'
    pred_benchmark = np.percentile(predictedAll, predpercentile)
    print("25% is {0} and 50% {1}".format(np.percentile(ydata_clean, 25), np.percentile(ydata_clean, 50)))
    # print("the benchmark at {0} percentile is {1}".format(predictedAll, pred_benchmark))
    for i in range(0, len(predictedAll)):
        if (predictedAll[i] >= pred_benchmark):
            predictedAll[i] = 1
        else:
            predictedAll[i] = 0

    # calculate errors
    counter = 0
    # error msg if diff sizes
    if (len(predictedAll) != len(ydata_reclass)):
        print("DIFF LENGTHS")
        print("length predicted {0}".format(len(predictedAll)))
        print("length reclass {0}".format(len(ydata_reclass)))
        return

    for i in range(0, len(ydata_reclass)):
        if (predictedAll[i] == ydata_reclass[i]):
            counter += 1

    return counter/len(ydata_reclass)


def predictErosionSingle(ry, rx, listofndvi, ybandnum, ybenchmark, predpercentile):
    '''
    predictErosion(ry, rx_vv, rx_vh, listofndvi, ybandnum, ybenchmark, predpercentile): uses random forest modeling to
        predict erosion, using the vv coherence band rx. The data above percentile predpercentile
        pixels are considered 'significant', to be compared with the erosion on the ry band, using ybenchmark.
    :param ry: the in-situ results
    :param rx: coherence product
    :param listofndvi: list of ndvi over the time period
    :param ybandnum: the product i was using for ry had multiple bands. For ease of use, this species each band
    :param ybenchmark: benchmark for what is considered erosion on the ry data
    :param predpercentile: anything above this percentile value is considered eroded
    :return:
    '''

    # 2 lambda functions used to transform the shape of the input into regression
    innerList = lambda ll, n: [ll[i][n] for i in range(0, len(ll))]
    transpose = lambda ll: [innerList(ll, i) for i in range(0, len(ll[0]))]

    print("Y benchmark of {0} and ypredicted benchmark of {1}".format(ybenchmark, predpercentile))

    # get resultant data

    yraster = gdal.Open(ry)
    yband = yraster.GetRasterBand(ybandnum)
    ydata = yband.ReadAsArray().flatten()

    # get vv data
    vvraster = gdal.Open(rx)
    vvband = vvraster.GetRasterBand(1)
    vvdata = (vvband.ReadAsArray()).flatten()

    ndvidatalist = []
    # get ndvi data
    for tile in listofndvi:
        ndviraster = gdal.Open(tile)
        ndviband = ndviraster.GetRasterBand(1)
        ndvidata = ndviband.ReadAsArray().flatten()
        print("The x of this band is {0}, y is {1}".format(ndviraster.RasterXSize, ndviraster.RasterYSize))
        ndvidatalist.append(ndvidata)
        print("AAA dirs {0}".format(len(ndvidata)))

    # get rid of nodata values.
    # get all nodata values
    removelist = []
    for i in range(0, len(vvdata)):
        if (vvdata[i] == nodata or ydata[i] == nodata):
            removelist.append(i)

    print("{0} values have had to be removed".format(len(removelist)))

    vvdata_clean = vvdata
    ydata_clean = ydata
    ndvidatalist_clean = ndvidatalist
    # remove all nodata values
    for i in removelist[::-1]:
        vvdata_clean = np.delete(vvdata_clean, i)
        ydata_clean = np.delete(ydata_clean, i)

        for j in range(0, len(ndvidatalist_clean)):
            ndvidatalist_clean[j] = np.delete(ndvidatalist_clean[j], i)

    del vvdata, ydata, ndvidatalist

    # print("New list lengths are {0} | {1} | {2} for x and y respectively".format(len(vvdata_clean), len(ydata_clean)))

    # prep the explanatory variables
    explanatory = ndvidatalist_clean
    explanatory.append(vvdata_clean)
    print("AAAAAAA {0}".format(np.shape(explanatory)))
    explanatory = transpose(explanatory)

    # print(explanatory)

    reg1 = RandomForestRegressor(max_depth = 2, random_state = 0)
    reg1.fit(explanatory, ydata_clean)
    # randomize the list
    # explanatory, ydata_clean = zip(*sorted(zip(explanatory, ydata_clean)))
    explanatory, ydata_clean = shuffle(explanatory, ydata_clean, random_state = 0)

    print("Partition at {0} for dataset {1}".format(int(np.round(7*(len(explanatory))/10)), np.shape(explanatory)))
    training = explanatory[:int(np.round(7*(len(explanatory))/10))]
    validation = explanatory[int(np.round(7*(len(explanatory))/10)) + 1:]
    predtrain = reg1.predict(training)
    predval = reg1.predict(validation)

    # print(predtrain)

    # double check some of the accuracy
    print("Correlation in the training dataset is {0}".format(stats.pearsonr(predtrain, ydata_clean[:int(np.round(7*(len(explanatory))/10))])))
    print("Correlation in the validation dataset is {0}".format(stats.pearsonr(predval, ydata_clean[int(np.round(7*(len(explanatory))/10))+ 1:])))

    # make a regressor off some of the training data
    predictedAll = reg1.predict(explanatory)

    # reclassify the ydata to 'erosion/no erosion'
    ydata_reclass = ydata_clean
    for i in range(0, len(ydata_reclass)):
        if (ydata_reclass[i] >= ybenchmark):
            ydata_reclass[i] = 1
        else:
            ydata_reclass[i] = 0

    # reclassify the predicted to 'erosion/no erosion'
    pred_benchmark = np.percentile(predictedAll, predpercentile)
    print("25% is {0} and 50% {1}".format(np.percentile(ydata_clean, 25), np.percentile(ydata_clean, 50)))
    # print("the benchmark at {0} percentile is {1}".format(predictedAll, pred_benchmark))
    for i in range(0, len(predictedAll)):
        if (predictedAll[i] >= pred_benchmark):
            predictedAll[i] = 1
        else:
            predictedAll[i] = 0

    # calculate errors
    counter = 0
    # error msg if diff sizes
    if (len(predictedAll) != len(ydata_reclass)):
        print("DIFF LENGTHS")
        print("length predicted {0}".format(len(predictedAll)))
        print("length reclass {0}".format(len(ydata_reclass)))
        return

    for i in range(0, len(ydata_reclass)):
        if (predictedAll[i] == ydata_reclass[i]):
            counter += 1

    return counter/len(ydata_reclass)




