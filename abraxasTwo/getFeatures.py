"""
cb, 09.07.2018

 - get normalized features from data-set

Inputs:
windowedData     := Input data-set. Array with shape windowedData[i][j, k], where i is the data window index, j is the
                time index and k refers to the sensors/data.

numCoeffs       := Number of dominant wavelet coefficients appended to the feature vector.

numFreqs        := Number of dominant frequencies appended to the feature vector.

normalize:      := When None the features are not normalized, else the features are normalized with the respective
                method. See abraxasTwo.normalize. Default stand.

Output:
featureMatrix   := Matrix filled with (optionally normalized) feature vectors.

normVal         := Normalization constants. In case:
                stand   -> [mean(x), standard_dev(x)]
                mean    -> [mean(x), min(x), max(x)]
                minmax  -> [min(x), max(x)].

"""

from abraxasTwo.extractFeatures import extractFeatures
import matplotlib.pyplot as plt
import numpy as np


def getFeatures(windowedData, numCoeffs, numFreqs, wvltLvl1=False, statFeat=True, wavelet='haar', dataQueue=False):

    featureMatrix = []

    for i in range(len(windowedData)):
        featureMatrix.append(extractFeatures(windowedData[i], numCoeffs=numCoeffs, numFreqs=numFreqs,
                                             wvltLvl1=wvltLvl1, statFeat=statFeat, wavelet=wavelet,
                                             dataQueue=dataQueue))

    return np.array(featureMatrix)


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet
    from abraxasTwo.windowDataSet import windowDataSet

    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2, startTimes=0,
                          stopTimes=10**9, selBnoData=[False, False, False], checkData=False, equalLength=False,
                          rawData=None)

    dataWindows, windowLabels = windowDataSet(dataSet, labels=[0, 1, 2, 3, 4, 1, 0], width=500, shift=500)
    features = getFeatures(dataWindows, numCoeffs=10, numFreqs=10)
    plt.plot(features[0])
    plt.show()

