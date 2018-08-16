"""
cb, 23.07.2018

 - DTW on RABE-data, user-identification

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from abraxasOne.helperFunctions import writeMatrixToCsvFile
from abraxasThree.classifierClass import AbraxasClassifier
from abraxas4.abraxasFrame import AbraxasFrame
from fastdtw import fastdtw
from abraxasOne.gaussFilter import gaussFilter

b = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=250, windowShift=50, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='none', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)

b.setWindowFunction(functionName='kaiser', alpha=0.25)
# b.plotWindowFunction()

b.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
b.selectSensorSubset(selectedSensors=[], sensorType='fr')
# b.selectSensorSubset(selectedSensors=[0], sensorType='ir')

b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=0)

b.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

b.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=2)

dataSet = b.readDataSet(checkData=False, equalLength=True)
# dataSet := Array with shape dataSet[i][j, k], where i refers to the i-th file loaded, k indicates the sensor and
#         j is the "time"-index.

wData, wLabels = b.windowSplitSourceDataTT(inputData=dataSet, inputLabels=np.array([0, 1, 2]))

wLabels = np.array(wLabels)

print("Number of windows, Igor: ", str(np.size(wLabels[wLabels == 0])))
print("Number of windows, Ankita: ", str(np.size(wLabels[wLabels == 1])))
print("Number of windows, Markus: ", str(np.size(wLabels[wLabels == 2])))

igor = []
ankita = []
markus = []

for i in range(len(wLabels)):
    if wLabels[i]==0:
        igor.append(wData[i])
    if wLabels[i]==1:
        ankita.append(wData[i])
    if wLabels[i]==2:
        markus.append(wData[i])

trainFrac = 1/2

iTrain = igor[0:int(trainFrac*len(igor))]
iTest = igor[int(trainFrac*len(igor))::]
aTrain = ankita[0:int(trainFrac*len(igor))]
aTest = ankita[int(trainFrac*len(igor))::]
mTrain = markus[0:int(trainFrac*len(igor))]
mTest = markus[int(trainFrac*len(igor))::]

testLabels = np.concatenate([np.zeros(len(iTest)), np.ones(len(aTest)), 2*np.ones(len(mTest))])

igorDistance = []
ankitaDistance = []
markusDistance = []
labels = []

print("Number of test windows, Igor: ", str(len(iTest)))
print("Number of test windows, Ankita: ", str(len(aTest)))
print("Number of test windows, Markus: ", str(len(mTest)))

test = np.concatenate([iTest, aTest, mTest])

for i in range(len(test)):
    testData = test[i]
    print(i/len(test))
    dOut = 10**20
    for k in range(len(iTrain)):
        d = 0
        for j in range(np.size(wData[0][0, ::])):
            distance, path = fastdtw(iTrain[k][::, j], testData[::, j], dist=euclidean)
            d += distance
        if d<dOut:
            dOut = d
    igorDistance.append(dOut/np.size(wData[0][0, ::]))
    dOut = 10**20
    for k in range(len(aTrain)):
        d = 0
        for j in range(np.size(wData[0][0, ::])):
            distance, path = fastdtw(aTrain[k][::, j], testData[::, j], dist=euclidean)
            d += distance
        if d<dOut:
            dOut = d
    ankitaDistance.append(dOut/np.size(wData[0][0, ::]))
    dOut = 10**20
    for k in range(len(mTrain)):
        d = 0
        for j in range(np.size(wData[0][0, ::])):
            distance, path = fastdtw(mTrain[k][::, j], testData[::, j], dist=euclidean)
            d += distance
        if d<dOut:
            dOut = d
    markusDistance.append(dOut/np.size(wData[0][0, ::]))
    labels.append(wLabels[i])

ankitaDistance = np.array(ankitaDistance)
markusDistance = np.array(markusDistance)
igorDistance = np.array(igorDistance)
labels = np.array(labels)
axis = np.linspace(0, len(labels)-1, len(labels))

plt.plot(axis, ankitaDistance, label='ankita', color='r')
writeMatrixToCsvFile(data=np.array([axis, ankitaDistance]).T, fileName="ankitaCostTrSet.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.plot(axis, igorDistance, label='igor', color='g')
writeMatrixToCsvFile(data=np.array([axis, igorDistance]).T, fileName="igorCostTrSet.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.plot(axis, markusDistance, label='markus', color='b')
writeMatrixToCsvFile(data=np.array([axis, markusDistance]).T, fileName="markusCostTrSet.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.stem(axis, 25*testLabels)
writeMatrixToCsvFile(data=np.array([axis, 25*labels]).T, fileName="labelsDTWTrSet.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.legend()
plt.show()

