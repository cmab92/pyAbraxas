"""
cb, 06.07.2018

Load raw data from .txt-file, clean errors and split sensor data.

Detailed:

Inputs:
fileName        := Name of target file.

nrIrSensors     := Number of infrared sensors. Default 10.

nrFrSensors     := Number of force sensors. Default 2.

interpolateBno  := Interpolate bno data, to get same number of data values as given for analog data. Default True.

tSample         := Sampling interval in seconds. Default 0.0165.

dirPath         := Destination file system path. Default is current working directory.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scinterp


def loadFile(fileName, nrIrSensors, nrFrSensors, interpolateBno=True, tSample=0.0165, dirPath=""):

    # load data from file:
    data = np.loadtxt(dirPath + fileName, delimiter=',', comments="%")
    cols = np.size(data[0, ::])
    numberOfAnalogSensors = nrIrSensors + nrFrSensors

    # ir data (clean transmission errors):
    irData = data[::, 0:nrIrSensors]
    irData[irData > 1023] = 1023
    irData[irData < 0] = 0

    # force Data (clean transmission errors):
    forceData = 1023 - data[::, nrIrSensors:numberOfAnalogSensors]
    forceData[forceData > 1023] = 1023
    forceData[forceData < 0] = 0

    # handle bno data (normalize):
    temp = data[data[::, cols-1] == 0]     # get lines with quatData
    quatData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+4)]/2**14  # normalization (14 bit) and scale as analog data
    temp = data[data[::, cols-1] == 1]     # get lines with lin acc data
    linAccData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+3)]/2**4  # arbitrary normalization
    temp = data[data[::, cols-1] == 2]     # get lines with ang vec data
    angVecData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+3)]/2**4  # arbitrary normalization

    # quat, linaAcc and angVec possibly of different length ...:
    bnoDataSize = np.min([np.size(quatData[::, 0]), np.size(linAccData[::, 0]), np.size(angVecData[::, 0])])
    quatData = quatData[:bnoDataSize, ::]
    linAccData = linAccData[:bnoDataSize, ::]
    angVecData = angVecData[:bnoDataSize, ::]

    tAxisBno = np.linspace(0, 3*tSample*bnoDataSize, bnoDataSize)
    tAxisAnalog = np.linspace(0, tSample*(np.size(irData[::, 0])-1), np.size(irData[::, 0]))

    if interpolateBno:
        quatDataTemp = []
        linAccDataTemp = []
        angVecDataTemp = []
        for i in range(4):
            tempData = quatData[::, 3]
            tempAxis = np.linspace(0, 3*tSample*bnoDataSize, bnoDataSize)
            dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+1), bnoDataSize+1)
                try:
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+2), bnoDataSize+2)
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            quatDataTemp.append(dataInterp(tAxisAnalog))
        quatData = np.array(quatDataTemp).T
        for i in range(3):
            tempData = linAccData[::, i]
            tempAxis = np.linspace(0, 3*tSample*bnoDataSize, bnoDataSize)
            dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+1), bnoDataSize+1)
                try:
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+2), bnoDataSize+2)
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            linAccDataTemp.append(dataInterp(tAxisAnalog))
        linAccData = np.array(linAccDataTemp).T
        for i in range(3):
            tempData = angVecData[::, i]
            tempAxis = np.linspace(0, 3*tSample*bnoDataSize, bnoDataSize)
            dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+1), bnoDataSize+1)
                try:
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3*tSample*(bnoDataSize+2), bnoDataSize+2)
                    dataInterp = scinterp.interp1d(tempAxis, tempData, kind='linear')
            angVecDataTemp.append(dataInterp(tAxisAnalog))
        angVecData = np.array(angVecDataTemp).T

    temp = []
    for i in range(nrIrSensors):
        temp2 = []
        temp2.append(tAxisAnalog)
        temp2.append(irData[::, i])
        temp.append(np.transpose(temp2))
    irData = temp
    temp = []
    for i in range(nrFrSensors):
        temp2 = []
        temp2.append(tAxisAnalog)
        temp2.append(forceData[::, i])
        temp.append(np.transpose(temp2))
    forceData = temp
    temp = []
    if interpolateBno:
        for i in range(3):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(linAccData[::, i])
            temp.append(np.transpose(temp2))
        linAccData = temp
        temp = []
        for i in range(3):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(angVecData[::, i])
            temp.append(np.transpose(temp2))
        angVecData = temp
        temp = []
        for i in range(4):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(quatData[::, i])
            temp.append(np.transpose(temp2))
        quatData = temp
    else:
        for i in range(3):
            temp2 = []
            temp2.append(tAxisBno)
            temp2.append(linAccData[::, i])
            temp.append(np.transpose(temp2))
        linAccData = temp
        temp = []
        for i in range(3):
            temp2 = []
            temp2.append(tAxisBno)
            temp2.append(angVecData[::, i])
            temp.append(np.transpose(temp2))
        angVecData = temp
        temp = []
        for i in range(4):
            temp2 = []
            temp2.append(tAxisBno)
            temp2.append(quatData[::, i])
            temp.append(np.transpose(temp2))
        quatData = temp
    return irData, forceData, quatData, linAccData, angVecData


if __name__ == '__main__':
    irData, forceData, quatData, linAccData, angVecData = loadFile(fileName="ankita.txt", nrIrSensors=10,
                                                                   nrFrSensors=2, dirPath="../")
    if 1:
        for i in range(10):
            plt.plot(irData[i][::, 0], irData[i][::, 1])
        plt.figure()
        for i in range(2):
            plt.plot(forceData[i][::, 0], forceData[i][::, 1])
        plt.figure()
        plt.plot(linAccData[0][::, 0], linAccData[0][::, 1])
        plt.plot(linAccData[1][::, 0], linAccData[1][::, 1])
        plt.plot(linAccData[2][::, 0], linAccData[2][::, 1])
        plt.title("linAcc")
        plt.figure()
        plt.plot(angVecData[0][::, 0], angVecData[0][::, 1])
        plt.plot(angVecData[1][::, 0], angVecData[1][::, 1])
        plt.plot(angVecData[2][::, 0], angVecData[2][::, 1])
        plt.figure()
        plt.plot(quatData[0][::, 0], quatData[0][::, 1])
        plt.plot(quatData[1][::, 0], quatData[1][::, 1])
        plt.plot(quatData[2][::, 0], quatData[2][::, 1])
        plt.plot(quatData[3][::, 0], quatData[3][::, 1])
        plt.show()