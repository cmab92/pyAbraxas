import numpy as np
import matplotlib.pyplot as plt
from abraxasOne.loadAndClean import loadAndClean


def readSeveralFiles(files, startTimes, stopTimes, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=True, checkData=True, useBno=False, interpBno=False, useForce=False, useIr=True, selectSensors=None):
    if equalLength:
        stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - startTimes) + startTimes
        stopTimes = [int(x) for x in stopTimes]

    if isinstance(selectSensors, (list, tuple, np.ndarray)):
        dataSet = []
        for i, element in enumerate(files):
            irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, interpolateBno=interpBno, tSample=0.0165, dirPath=path)
            dataSetTemp = []
            if useIr:
                for j in range(len(selectSensors)):
                    dataSetTemp.append(irData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
            if useForce:
                for j in range(numberOfForceSensors):
                    dataSetTemp.append(forceData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
            if useBno:
                for j in range(4):
                    dataSetTemp.append(quatData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
                for j in range(3):
                    dataSetTemp.append(linAccData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
                for j in range(3):
                    dataSetTemp.append(angVecData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
            sensorData = np.array(dataSetTemp).T
            dataSet.append(sensorData)
    else:
        dataSet = []
        for i, element in enumerate(files):
            irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, numberOfIrSensors, numberOfForceSensors, interpolateBno=interpBno, tSample=0.0165, dirPath=path)
            dataSetTemp = []
            if useIr:
                for j in range(len(irData)):
                    dataSetTemp.append(irData[j][startTimes[i]:stopTimes[i], 1])
            if useForce:
                for j in range(numberOfForceSensors):
                    dataSetTemp.append(forceData[j][startTimes[i]:stopTimes[i], 1])
            if useBno:
                for j in range(4):
                    dataSetTemp.append(quatData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
                for j in range(3):
                    dataSetTemp.append(linAccData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
                for j in range(3):
                    dataSetTemp.append(angVecData[int(selectSensors[j])][startTimes[i]:stopTimes[i], 1])
            sensorData = np.array(dataSetTemp).T
            dataSet.append(sensorData)

    if checkData:
        for i, element in enumerate(files):
            for j in range(len(dataSet[i][0,::])):
                plt.plot(dataSet[i][::,j], label=str(j))
                plt.title(files[i])
            plt.legend()
            plt.show()
    return dataSet