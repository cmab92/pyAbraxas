"""
cb, 07.07.2018

 - slice data-set to windows and label

Inputs:
dataSet         := Input data-set given as dataSet[i][j, k], where i refers to the i-th file loaded, k indicates the sensor
                and j is the "time"-index.

labels          := Labels for dataSet, given in the order of dataSet (dataSet[i] belongs to labels[i]).

width           := Width of data windows. Default 100. Unit is number of samples.

shift           := Specifies time shift between windows. Given in samples.

window          := Specifies window function. See abraxasTwo.miscFunc, applyWindow. Default 'tukey'.

alpha           := Shape parameter for window function. See abraxasTwo.miscFunc, applyWindow. Default 0.1.

sampleT         := Enable window function check. Plots time function and frequency response of window function if the
                sampling interval duration is given (in seconds). Default None.

enaCheck        := Enable data check. Plots the data of dataSet[i]. Default False.

enaWindowC      := Enable window check. Plots the created data windows. Default False.

Outputs;
windowedData    := Array with windowed Data. windowedData

windowLabels    := Data labels for data windows. in order of dataWindow.

"""

import numpy as np
import matplotlib.pyplot as plt
from abraxasTwo.miscFunc import applyWindow


def windowDataSet(dataSet, labels, width, shift, window='tukey', alpha=0.1, sampleT=None, enaCheck=False
                  , enaWindowC=False):
    if len(dataSet) != len(labels):
        print("Dimension mismatch: labels < - > dataSet")

    windowedData = []
    windowLabels = []
    for k in range(len(dataSet)):

        data = dataSet[k]

        numOfSensors = len(data[::].T)
        if enaCheck:
            plt.figure()
            for i in range(numOfSensors):
                plt.plot(data[::, i], label=i)
            plt.legend()
            plt.title('Check selected data. Label := ' + str(labels[k]))
            plt.show()

        minDataLength = []
        for i in range(numOfSensors):
            minDataLength.append(np.size(data[::, i]))
        minDataLength = np.min(minDataLength)
        tempNumWindows = int((minDataLength-width)/shift+1)
        for i in range(tempNumWindows):
            windowedDataTemp = []
            for j in range(numOfSensors):
                dataTemp = data[i*shift:(i*shift+width), j]
                windowedDataTemp.append(applyWindow(dataTemp, window=window, alpha=alpha, samplingPeriod=sampleT))
            windowedDataTemp = np.transpose(windowedDataTemp)
            windowedData.append(windowedDataTemp)
            windowLabels.append(labels[k])
            if enaWindowC:
                plt.plot(windowedDataTemp)
                plt.title("Label:=" + str(labels[k]))
                plt.show()
    return windowedData, windowLabels


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet
    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2,
                          selBnoData=[True, True, True], checkData=False, equalLength=False, rawData=None)

    dataWindows, windowLabels = windowDataSet(dataSet, labels=[0, 1, 2, 3, 4, 1, 0], width=1000, shift=10)
    plt.plot(dataWindows[0][100:200, 0])
    plt.show()

