"""
cb, 09.07.2018

 - get normalized features from data-set

Inputs:
windowedData     := Input data-set. Array with shape windowedData[i][j, k], where i is the data window index, j is the
                time index and k refers to the sensors/data.

numCoeffs       := Number of dominant wavelet coefficients appended to the feature vector.

numFreqs        := Number of dominant frequencies appended to the feature vector.

normalize:      := When None the features are not normalized, else the features are normalized with the respective
                method:
                stand   -> standardization, x_new = (x - mean(x))/standard_dev(x)
                mean    -> mean normalization, x_new = (x - mean(x))/(max(x) - min(x))
                minmax  -> rescaling, (x - min(x))/(max(x) - min(x)).
                Default stand.

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


def getFeatures(windowedData, numCoeffs, numFreqs, normalize='stand'):

    featureMatrix = []

    for i in range(len(windowedData)):
        featureMatrix.append(extractFeatures(windowedData[i], numCoeffs=numCoeffs, numFreqs=numFreqs))

    featureMatrix = np.array(featureMatrix)

    if normalize is not None:
        if normalize == 'stand':
            mue = []
            sigma = []
            for i in range(len(featureMatrix[0][::])):
                mue.append(np.mean(featureMatrix[::, i]))
                sigma.append(np.std(featureMatrix[::, i]))
                if sigma[i] == 0:
                    sigma[i] = np.mean(featureMatrix[0, i])*10**6 + 10**6
                featureMatrix[::, i] = (featureMatrix[::, i] - mue[i])/sigma[i]
            normVal = [mue, sigma]
        elif normalize == 'mean':
            mue = []
            minVal = []
            maxVal = []
            for i in range(len(featureMatrix[0][::])):
                mue.append(np.mean(featureMatrix[::, i]))
                minVal.append(np.min(featureMatrix[::, i]))
                maxVal.append(np.max(featureMatrix[::, i]))
                if (maxVal[i] - minVal[i]) == 0:
                    maxVal[i] = np.abs(minVal[i]*10**6)
                featureMatrix[::, i] = (featureMatrix[::, i] - mue[i])/(maxVal[i] - minVal[i])
            normVal = [mue, minVal, maxVal]
        elif normalize == 'minmax':
            minVal = []
            maxVal = []
            for i in range(len(featureMatrix[0][::])):
                minVal.append(np.min(featureMatrix[::, i]))
                maxVal.append(np.max(featureMatrix[::, i]))
                if (maxVal[i] - minVal[i]) == 0:
                    maxVal[i] = np.abs(minVal[i]*10**6)
                featureMatrix[::, i] = (featureMatrix[::, i] - minVal[i])/(maxVal[i] - minVal[i])
            normVal = [minVal, maxVal]
        else:
            mue = []
            sigma = []
            for i in range(len(featureMatrix[0][::])):
                mue.append(np.mean(featureMatrix[::, i]))
                sigma.append(np.std(featureMatrix[::, i]))
                if sigma[i] == 0:
                    sigma[i] = np.mean(featureMatrix[0, i])*10**1 + 10**6
                featureMatrix[::, i] = (featureMatrix[::, i] - mue[i])/sigma[i]
            normVal = [mue, sigma]
        normVal = np.array(normVal)
    else:
        normVal = None

    return featureMatrix, normVal


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet
    from abraxasTwo.windowDataSet import windowDataSet

    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2,
                          selBnoData=[False, False, False], checkData=False, equalLength=False, rawData=None)

    dataWindows, windowLabels = windowDataSet(dataSet, labels=[0, 1, 2, 3, 4, 1, 0], width=500, shift=500)
    features, normVals = getFeatures(dataWindows, numCoeffs=10, numFreqs=10, normalize='stand')
    plt.plot(features[0])
    plt.show()

