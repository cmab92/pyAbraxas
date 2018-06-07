########################################################################################################################
## cb, 20.04.18
##
## general info:
## load, clean and split data
##
## function inputs:
## fileName: output file name
## numberOfIrSensors: ...
## numberOfForceSensors: ...
## tSample: smapling time (analog values) ... assuming tSample_bno = 3*tSample_analog
## dirPath: input file directory
## normAnalog: normalization of analog data (boolean)
##
## functions output:
## irData, forceData: data[i][j,k] -> i-th sensor, jth datapoint (k=0 -> time, k=1 -> value)
## quatData, linAccData, angVecData: ...
##
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
def loadAndClean(fileName, numberOfIrSensors, numberOfForceSensors, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/"):
    # load data:
    data = np.loadtxt(dirPath + fileName, delimiter=',', comments="%") # data: data[i,j]:= i-th data point of j-th sensor
    rows = np.size(data[::,0])
    cols = np.size(data[0,::])
    numberOfAnalogSensors = numberOfIrSensors + numberOfForceSensors
    # ir data (normalize and clean transmission errors):
    irData = data[::, 0:numberOfIrSensors]
    irData[irData > 1023] = 1023
    irData[irData < 0] = 0
    forceData = 1 - data[::, numberOfIrSensors:numberOfAnalogSensors]
    forceData[forceData > 1023] = 1023
    forceData[forceData < 0] = 0
    # force Data (normalize and clean transmission errors):
    # handle bno data (normalize):
    temp = data[data[::, cols-1]==0]     # get lines with quatData
    quatData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+4)] # normalization (14 bit)
    quatData[quatData>1] = 1
    quatData[quatData<-1] = -1
    norm = 10 # normalization of linAcc and angVec
    upperLim = 1000
    temp = data[data[::, cols-1]==1]     # get lines with lin acc data
    linAccData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+3)] # arbitrary normalization
    linAccData[linAccData>upperLim/norm] = 0
    temp = data[data[::, cols-1]==2]     # get lines with ang vec data
    angVecData = temp[::, numberOfAnalogSensors:(numberOfAnalogSensors+3)] # arbitrary normalization
    angVecData[angVecData>upperLim/norm] = 0
    # quat, linaAcc and angVec possibly of different length ...:
    bnoDataSize = np.min([np.size(quatData[::,0]), np.size(linAccData[::,0]), np.size(angVecData[::,0])])
    quatData = quatData[:bnoDataSize,::]
    linAccData = linAccData[:bnoDataSize,::]
    angVecData = angVecData[:bnoDataSize,::]
    # generate time axis
    tAxisAnalog = np.linspace(0, tSample*np.size(irData[::,0]), np.size(irData[::,0]))
    tAxisBno = np.linspace(0, 3*tSample*bnoDataSize, bnoDataSize)
    temp = []
    for i in range(numberOfIrSensors):
        temp2 = []
        temp2.append(tAxisAnalog)
        temp2.append(irData[::,i])
        temp.append(np.transpose(temp2))
    irData = temp
    temp = []
    for i in range(numberOfForceSensors):
        temp2 = []
        temp2.append(tAxisAnalog)
        temp2.append(forceData[::,i])
        temp.append(np.transpose(temp2))
    forceData = temp
    temp = []
    for i in range(3):
        temp2 = []
        temp2.append(tAxisBno)
        temp2.append(linAccData[::,i])
        temp.append(np.transpose(temp2))
    linAccData = temp
    temp = []
    for i in range(3):
        temp2 = []
        temp2.append(tAxisBno)
        temp2.append(angVecData[::,i])
        temp.append(np.transpose(temp2))
    angVecData = temp
    temp = []
    for i in range(4):
        temp2 = []
        temp2.append(tAxisBno)
        temp2.append(quatData[::,i])
        temp.append(np.transpose(temp2))
    quatData = temp
    return irData, forceData, quatData, linAccData, angVecData

if __name__ == '__main__':
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(fileName="_20185161226.txt", numberOfIrSensors=10, numberOfForceSensors=4)
    for i in range(10):
        plt.plot(irData[i][::,0], irData[i][::,1])
    plt.figure()
    for i in range(4):
        plt.plot(forceData[i][::,0], forceData[i][::,1])
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