import numpy as np
import random

from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractFeatures import extractFeatures


def splitDataTrainTest(files, start, stop, fileLabels, windowWidth=100, windowShift=10, numDomCoeffs=10, numDomFreqs=10, trainFrac=2/3, statFeat=True, shuffleData=False, checkData=False):
    usedSensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10,
                               numberOfForceSensors=2, equalLength=True, checkData=checkData, useForce=True, useBno=True,
                               useIr=True, interpBno=True, selectSensors=usedSensors)
    dataWindows = []
    numberOfWindows = []
    for i in range(len(dataSet)):
        windows, numOfWindows = sliceAndWindowV3(data=dataSet[i], windowWidth=windowWidth, windowShift=windowShift, enaCheck=False, window='tukey', alpha=0.1, enaCWF=0)
        dataWindows.append(windows)
        numberOfWindows.append(numOfWindows)
    dataWindows = np.array(dataWindows)
    numberOfWindows = np.array(numberOfWindows)
    features = []
    labels = []
    trainingFeatures = []
    testFeatures = []
    trainingLabels = []
    testLabels = []
    for i in range(len(dataWindows)):
        index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
        if shuffleData:
            random.shuffle(index)
        for j in range(numberOfWindows[i]):
            f = extractFeatures(dataWindows[i][int(index[j])], numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, statFeat=statFeat, wavelet='haar')
            if j>int(trainFrac*numberOfWindows[i]-2):
                testFeatures.append(f.T)
                testLabels.append(fileLabels[i])
            else:
                trainingFeatures.append(f.T)
                trainingLabels.append(fileLabels[i])
    trainingFeatures = np.array(trainingFeatures)
    testFeatures = np.array(testFeatures)
    mean = []
    std_dev = []
    for i in range(np.size(trainingFeatures[0, ::])):
        x = trainingFeatures[::, i]
        mean.append(sum(x) / len(x))
        std_dev.append((1 / len(x) * sum([(x_i - mean[i]) ** 2 for x_i in x])) ** 0.5)
        trainingFeatures[::, i] = (trainingFeatures[::, i] - mean[i]) / std_dev[i]
        testFeatures[::, i] = (testFeatures[::, i] - mean[i]) / std_dev[i]
    return trainingFeatures, testFeatures, trainingLabels, testLabels, mean, std_dev