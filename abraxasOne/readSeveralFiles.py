import numpy as np
import matplotlib.pyplot as plt
from abraxasOne.loadAndClean import loadAndClean
def readSeveralFiles(files, startTimes, stopTimes, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=True, checkData=True, selectSensors=None):
    if equalLength:
        stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - startTimes) + startTimes
        stopTimes = [int(x) for x in stopTimes]

    if isinstance(selectSensors, (list, tuple, np.ndarray)):
        dataSet = []
        for i, element in enumerate(files):
            irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, tSample=0.0165, dirPath=path)
            irDataTemp = []
            for j in range(len(selectSensors)):
                irDataTemp.append(irData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
            irData = np.array(irDataTemp).T
            dataSet.append(irData)
    else:
        dataSet = []
        for i, element in enumerate(files):
            irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, tSample=0.0165, dirPath=path)
            irDataTemp = []
            for j in range(len(irData)):
                irDataTemp.append(irData[j][startTimes[i]:stopTimes[i], 1])
            irData = np.array(irDataTemp).T
            dataSet.append(irData)

    if checkData:
        for i, element in enumerate(files):
            for j in range(len(dataSet[i][0,::])):
                plt.plot(dataSet[i][::,j], label=str(j))
                plt.title(files[i])
            plt.legend()
            plt.show()
    return dataSet