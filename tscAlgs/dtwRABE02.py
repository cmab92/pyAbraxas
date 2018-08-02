"""
cb, 23.07.2018

 - DTW on RABE-data, binary acitivity recognition

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from abraxasOne.helperFunctions import writeMatrixToCsvFile
from abraxasThree.classifierClass import AbraxasClassifier
from fastdtw import fastdtw
from abraxasOne.gaussFilter import gaussFilter

b = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=150, windowShift=150, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='none', kernel=0, trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)
b.setWindowFunction(functionName='tukey', alpha=0.15)
# b.plotWindowFunction()

b.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
b.selectSensorSubset(selectedSensors=[], sensorType='fr')
b.selectSensorSubset(selectedSensors=[0, 1, 2, 4, 6, 8], sensorType='ir')
b.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=1500, label=0,
               className="walking")
b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=0)

b.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=0)
b.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=0)

b.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=0)
b.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=0)
b.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=300, stopTime=1700, label=0)

b.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=0)

b.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=0)

b.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=0)

b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=0)

b.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=1,
               className="not walking")
b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=1)
b.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=1)
b.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=1)
b.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=1)
b.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=1)
b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=1)
b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=1)
b.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=1)
b.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
b.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
b.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
b.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)

dataSet = b.readDataSet(checkData=False, equalLength=True)
# dataSet := Array with shape dataSet[i][j, k], where i refers to the i-th file loaded, k indicates the sensor and
#         j is the "time"-index.

wData, wLabels = b.windowSplitSourceDataTT(inputData=dataSet, inputLabels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                                    1, 1]))

wLabels = np.array(wLabels)

print(wLabels)

toggleI = 1
toggleA = 1

print("Number of windows, walk: ", str(np.size(wLabels[wLabels == 0])))
print("Number of windows, no walk: ", str(np.size(wLabels[wLabels == 1])))

for i in range(len(wLabels)):
    if wLabels[i] == 0:
        if toggleI == 1:
            toggleI = 0
            walkBase = wData[i]
    elif wLabels[i] == 1:
        if toggleA == 1:
            toggleA = 0
            noWalkBase = wData[i]

walkDistance = []
noWalkDistance = []
labels = []

for i in range(len(wLabels)):
    d = 0
    for j in range(np.size(wData[0][0, ::])):
        distance, path = fastdtw(walkBase[::, j], wData[i][::, j], dist=euclidean)
        d += distance
    walkDistance.append(d/np.size(wData[0][0, ::]))
    d = 0
    for j in range(np.size(wData[0][0, ::])):
        distance, path = fastdtw(noWalkBase[::, j], wData[i][::, j], dist=euclidean)
        d += distance
    noWalkDistance.append(d/np.size(wData[0][0, ::]))
    labels.append(wLabels[i])

noWalkDistance = np.array(noWalkDistance)
walkDistance = np.array(walkDistance)
labels = np.array(labels)
axis = np.linspace(0, len(labels)-1, len(labels))

plt.plot(axis, noWalkDistance, label='walk', color='r')
writeMatrixToCsvFile(data=np.array([axis, noWalkDistance]).T, fileName="walkCost250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.plot(axis, walkDistance, label='no walk', color='g')
writeMatrixToCsvFile(data=np.array([axis, walkDistance]).T, fileName="walkCost250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.stem(axis, 25*labels)
plt.legend()
plt.show()

