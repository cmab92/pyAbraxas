import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import peakutils as pu
from statsmodels import robust

## load data
from abraxasOne.loadAndClean import loadAndClean
files = ["ankita.txt", "chris_asymm.txt", "chris_c.txt", "chris_pos2.txt", "igor.txt", "igor2.txt", "ankita_pos2_lrRl.txt"]
start = np.array([250, 200, 100, 100, 100, 800, 100])
stop = np.array([1750, 1400, 1500, 1700, 3500, 6400, 2700])

def autoCorr(series, startLagFrom=0, lagUpTo=1):
    y = []
    for i in range(lagUpTo):
        y.append(pd.Series.autocorr(pd.Series(series), lag=i))
    x = np.linspace(startLagFrom, lagUpTo, (lagUpTo - startLagFrom))
    return x, np.array(y)[startLagFrom:]

corrTotal = []
for k, element in enumerate(files):
    dir = element
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(dir, 10, 2, tSample = 0.0165, dirPath = "")

    X = []
    autoCorrX = []
    corrX = []
    autoCorrPeakPosOfDataSet = []
    autoCorrPeakValOfDataSet = []
    begin = start[k]
    end = stop[k]
    for i in range(len(irData)):
        data = np.abs(irData[i][begin:end,1])
        autoCorrAxis, autoCorrData = autoCorr(series=data, startLagFrom=25, lagUpTo=200)
        indices = pu.indexes(autoCorrData, thres=0.1 / max(autoCorrData), min_dist=1)
        autoCorrPeak = np.array([autoCorrAxis[np.max(autoCorrData[indices])==autoCorrData], np.max(autoCorrData[indices])])
        autoCorrPeakPosOfDataSet.append(autoCorrPeak[0])
        autoCorrPeakValOfDataSet.append(autoCorrPeak[1])
        #plt.scatter(autoCorrPeak[0], autoCorrPeak[1], edgecolors='r')
        #plt.plot(autoCorrAxis, autoCorrData)
        #plt.show()
        autoCorrX.append(autoCorrData)
        corrXtemp = []
        for j in range(len(irData)):
            data2 = irData[j][begin:end,1]
            corrXtemp.append(pd.Series(data).corr(other=pd.Series(data2)))
        corrX.append(corrXtemp)
        #plt.plot(autoCorrData)

    autoCorrPeakPosOfDataSet = np.array(autoCorrPeakPosOfDataSet)
    autoCorrPeakValOfDataSet = np.array(autoCorrPeakValOfDataSet)
    #plt.hist(autoCorrPeakPosOfDataSet,color='b')
    #plt.hist(autoCorrPeakPosOfDataSet[np.argmax(autoCorrPeakValOfDataSet)],color='r')
    autoCorrPeakPosOfDataSet = int(np.mean(autoCorrPeakPosOfDataSet))
    for i in range(len(irData)):
        data = irData[i][begin:end,1]
        data = data[:int(len(data)-(len(data)%autoCorrPeakPosOfDataSet))]
        numberOfWindows = int(len(data) / autoCorrPeakPosOfDataSet)
        meanOfSensor = np.reshape(data, [numberOfWindows, autoCorrPeakPosOfDataSet])
        for j in range(numberOfWindows):
            plt.plot(data[:autoCorrPeakPosOfDataSet])
            plt.plot(meanOfSensor[j,::])

        meanOfSensor = np.mean(meanOfSensor, 0)
        plt.show()
        plt.figure()
        plt.plot(meanOfSensor)
        #plt.plot(meanOfSensor)
        plt.show()
    plt.show()
    corrX = np.array(corrX)
    #plt.title(dir)
    corrTotal.append(corrX)
    plt.show()

corrTotal = np.array(corrTotal)
corrTotal = np.sum(corrTotal,0)/len(corrTotal)
