import numpy as np
import matplotlib.pyplot as plt
from abraxasOne.loadAndClean import loadAndClean
def readSeveralFiles(files, startTimes, stopTimes, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=True, checkData=True):
    if equalLength:
        stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - startTimes) + startTimes
        stopTimes = [int(x) for x in stopTimes]

    if checkData:
        for i, element in enumerate(files):
            irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, tSample=0.0165, dirPath=path)
            for j in range(len(irData)):
                plt.plot(irData[j][startTimes[i]:stopTimes[i], 1], label=str(j))
                plt.title(files[i])
            plt.legend()
            plt.show()

    dataSet = []
    for i, element in enumerate(files):
        irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, tSample=0.0165, dirPath=path)
        for j in range(len(irData)):
            irData[j] = irData[j][startTimes[i]:stopTimes[i], 1]
        irData = np.array(irData).T
        dataSet.append(irData)
    return dataSet