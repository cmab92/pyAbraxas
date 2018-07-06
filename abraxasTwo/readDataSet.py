"""
cb, 04.07.2018

 - read dataset from .txt-files

 Detailed:
 Read complete ir-, force- and bno-data (or subsets of this data) from .txt-files and combine to a dataset consisting
 of labeled training and test data. Optionally the resulting dataset is stored.

Inputs:
fileNames       := List or 1d-array with names of files to be read.

filePath        := Path to the files to be read.

startTimes      := List or 1d-array with starting points (in samples) from which the respective file is read. The
                respective file means the file with the respective index (len(startTimes) should be equal to
                len(fileName)). Default [0, 0, ..., 0].


stopTimes       := List or 1d-array with stopping points (in samples) until which the respective file is read, else
                same as startTimes. Default [-1, -1, ..., -1].

nrIrSensors     := Number of infrared-sensors given in the files. Default 10.

nrFrSensors     := Number of force-sensors given in the files. Default 2.

selIrSensors    := List of infrared sensors to be selected. If None, all are selected. Default None.

selFrSensors    := List of force sensors to be selected. If None, all are selected. Default None.

selBnoData      := List of binary values [b_0, b_1, b_2], where b_0 (de)selects quaternion data, b_1 (de)selects linear
                acceleration data and b_2 (de)selects angular velocity data. Default [1, 1, 1].

checkData       := Plot each file, to check start- and stop-times. Default False.

equalLength     := Limits the file length to the mininum file length among all given. Default False.

rawData         := List of boolean, specifying whether the respective file contains raw data, or preprocessed data.
                Default [True, True, ..., True].

Output:
dataSet         := Array with shape dataSet[i][j, k], where i refers to the i-th file loaded, j indicates the sensor
                and k is the "time"-index.

"""

import numpy as np
import matplotlib.pyplot as plt
from abraxasTwo.loadFile import loadFile


def readDataSet(fileNames, filePath, startTimes=None, stopTimes=None, nrIrSensors=10, nrFrSensors=2, selIrSensors=None,
                selFrSensors=None, selBnoData=None, checkData=False, equalLength=False, rawData=None):

    if startTimes is None:
        startTimes = np.zeros(len(fileNames))
    if stopTimes is None:
        stopTimes = -np.ones(len(fileNames))
    if selIrSensors is None:
        selIrSensors = np.linspace(0, (nrIrSensors - 1), nrIrSensors)
    if selFrSensors is None:
        selFrSensors = np.linspace(0, (nrFrSensors - 1), nrFrSensors)
    if selBnoData is None:
        selBnoData = []
        for i in range(len(fileNames)):
            selBnoData.append(True)
    if rawData is None:
        rawData = []
        for i in range(len(fileNames)):
            rawData.append(True)
    if equalLength:
        stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - startTimes) + startTimes
        stopTimes = [int(x) for x in stopTimes]

    dataSet = []
    for i, element in enumerate(fileNames):
        irData, forceData, quatData, linAccData, angVecData = loadFile(element, nrIrSensors, nrFrSensors,
                                                                           interpolateBno=True, dirPath=filePath)
        dataSetTemp = []
        for j in range(len(selIrSensors)):
                dataSetTemp.append(irData[int(selIrSensors[j])][int(startTimes[i]):int(stopTimes[i]), 1])
        for j in range(len(selFrSensors)):
                dataSetTemp.append(forceData[int(selFrSensors[j])][int(startTimes[i]):int(stopTimes[i]), 1])
        if selBnoData[0]:
            for j in range(4):
                dataSetTemp.append(quatData[j][int(startTimes[i]):int(stopTimes[i]), 1])
        if selBnoData[1]:
            for j in range(3):
                dataSetTemp.append(linAccData[j][int(startTimes[i]):int(stopTimes[i]), 1])
        if selBnoData[2]:
            for j in range(3):
                dataSetTemp.append(angVecData[j][int(startTimes[i]):int(stopTimes[i]), 1])
        sensorData = np.array(dataSetTemp).T
        dataSet.append(sensorData)

    if checkData:
        for i, element in enumerate(fileNames):
            for j in range(len(dataSet[i][0, ::])):
                plt.plot(dataSet[i][::, j], label=str(j))
                plt.title(fileNames[i])
            plt.legend()
            plt.show()

    return dataSet


if __name__ == '__main__':
    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2, selBnoData=[False, False, False],
                checkData=True, equalLength=False, rawData=None)

