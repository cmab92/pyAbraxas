"""
cb, 23.07.2018

 - DTW on RABE-data, user-identification

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from abraxasOne.helperFunctions import writeMatrixToCsvFile
from abraxasThree.classifierClass import AbraxasClassifier
from fastdtw import fastdtw
from abraxasOne.gaussFilter import gaussFilter

b = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=250, windowShift=250, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='none', kernel=0, trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)
b.setWindowFunction(functionName='tukey', alpha=0.5)
# b.plotWindowFunction()

b.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
b.selectSensorSubset(selectedSensors=[], sensorType='fr')
b.selectSensorSubset(selectedSensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], sensorType='ir')

# b.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=1500, label=0,
#            className="walking")
b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=0)

# b.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1)
b.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

b.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=2)

dataSet = b.readDataSet(checkData=False, equalLength=True)
# dataSet := Array with shape dataSet[i][j, k], where i refers to the i-th file loaded, k indicates the sensor and
#         j is the "time"-index.

wData, wLabels = b.windowSplitSourceDataTT(inputData=dataSet, inputLabels=np.array([0, 1, 2]))

wLabels = np.array(wLabels)

print(wLabels)

toggleI = 1
toggleA = 1
toggleM = 1

print("Number of windows, Igor: ", str(np.size(wLabels[wLabels == 0])))
print("Number of windows, Ankita: ", str(np.size(wLabels[wLabels == 1])))
print("Number of windows, Markus: ", str(np.size(wLabels[wLabels == 2])))

for i in range(len(wLabels)):
    if wLabels[i] == 0:
        if toggleI == 1:
            toggleI = 0
            igorBase = wData[i]
    elif wLabels[i] == 1:
        if toggleA == 1:
            toggleA = 0
            ankitaBase = wData[i]
    elif wLabels[i] == 2:
        if toggleM == 1:
            toggleM = 0
            markusBase = wData[i]

igorDistance = []
ankitaDistance = []
markusDistance = []
labels = []

for i in range(len(wLabels)):
    d = 0
    for j in range(np.size(wData[0][0, ::])):
        distance, path = fastdtw(igorBase[::, j], wData[i][::, j], dist=euclidean)
        d += distance
    igorDistance.append(d/np.size(wData[0][0, ::]))
    d = 0
    for j in range(np.size(wData[0][0, ::])):
        distance, path = fastdtw(ankitaBase[::, j], wData[i][::, j], dist=euclidean)
        d += distance
    ankitaDistance.append(d/np.size(wData[0][0, ::]))
    d = 0
    for j in range(np.size(wData[0][0, ::])):
        distance, path = fastdtw(markusBase[::, j], wData[i][::, j], dist=euclidean)
        d += distance
    markusDistance.append(d/np.size(wData[0][0, ::]))
    labels.append(wLabels[i])

ankitaDistance = np.array(ankitaDistance)
markusDistance = np.array(markusDistance)
igorDistance = np.array(igorDistance)
labels = np.array(labels)
axis = np.linspace(0, len(labels)-1, len(labels))

plt.plot(axis, ankitaDistance, label='ankita', color='r')
writeMatrixToCsvFile(data=np.array([axis, ankitaDistance]).T, fileName="ankitaCost250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.plot(axis, igorDistance, label='igor', color='g')
writeMatrixToCsvFile(data=np.array([axis, igorDistance]).T, fileName="igorCost250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.plot(axis, markusDistance, label='markus', color='b')
writeMatrixToCsvFile(data=np.array([axis, markusDistance]).T, fileName="markusCost250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.stem(axis, 25*labels)
writeMatrixToCsvFile(data=np.array([axis, 25*labels]).T, fileName="labelsDTW250.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
plt.legend()
plt.show()

