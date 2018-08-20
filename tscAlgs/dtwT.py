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
from tscAlgs.triang import triang
from tscAlgs.dtwImp02 import dtwImp02

from abraxas4.abraxasFrame import AbraxasFrame
b = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=250, windowShift=50, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='none', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)

b.setWindowFunction(functionName='rect', alpha=0.25)
# b.plotWindowFunction()

b.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
b.selectSensorSubset(selectedSensors=[], sensorType='fr')
b.selectSensorSubset(selectedSensors=[0, 1, 2, 3, 5, 7, 9], sensorType='ir')

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

#for i in range(len(wLabels)):
#    wData[i] = wData[i]*(1+0*np.random.random([250, 1]))

for i in range(len(wLabels)):
    if wLabels[i]==0:
        igor.append(wData[i])
    if wLabels[i]==1:
        ankita.append(wData[i])
    if wLabels[i]==2:
        markus.append(wData[i])

sample0 = igor[0]
sample1 = ankita[0]
sample2 = markus[0]
refSample = [sample0, sample1, sample2]

ref0 = [0, 0]

d = 0
for j in range(np.size(wData[0][0, ::])):
    # distance = dtwImp02(sample1[::, j], sample0[::, j])
    distance, path = fastdtw(sample1[::, j], sample0[::, j], dist=euclidean)
    d += distance
d01 = d/np.size(wData[0][0, ::])
ref1 = [0, d01]

d = 0
for j in range(np.size(wData[0][0, ::])):
    # distance = dtwImp02(sample2[::, j], sample0[::, j])
    distance, path = fastdtw(sample2[::, j], sample0[::, j], dist=euclidean)
    d += distance
d20 = d/np.size(wData[0][0, ::])
d = 0
for j in range(np.size(wData[0][0, ::])):
    # distance = dtwImp02(sample2[::, j], sample1[::, j])
    distance, path = fastdtw(sample2[::, j], sample1[::, j], dist=euclidean)
    d += distance
d21 = d/np.size(wData[0][0, ::])
from tscAlgs.icPoints import icPoints
ref2x, ref2y, dummy, dummy = icPoints(d=np.array([d20, d21]), ref=np.array([ref0, ref1]))
ref2 = [ref2x, ref2y]

refCoord = [ref0, ref1, ref2]
# refCoord = [[0, 25], [-25, 25], [50, 0]]

fig, ax = plt.subplots()
box = 150
'''
ax.add_artist(plt.Circle(xy=refCoord[0], radius=d20, alpha=0.1, edgecolor='k'))
ax.add_artist(plt.Circle(xy=refCoord[1], radius=d21, alpha=0.1, edgecolor='k'))

plt.scatter(refCoord[0][0], refCoord[0][1], c='r', marker='x')
plt.scatter(refCoord[1][0], refCoord[1][1], c='r', marker='x')
plt.scatter(refCoord[2][0], refCoord[2][1], c='r', marker='x')

plt.xlim([-box, box])
plt.ylim([-box, box])
'''
# plt.scatter(ic[0], ic[1], marker='x', c='g')

#fig, ax = plt.subplots()

#for i in range(10):
#    igor.append(igor[0]+(i+1)*0.01*np.random.random([250, 1]))

igor.append(markus[0]/2+ankita[0]/2)
#igor.append(igor[0]+0.1*np.random.random([250, 1]))
#igor.append(igor[2])
lim = 60
for k in range(lim):
    print(k/(lim - 1))
    if k<20:
        test = markus[k]
        colorLabel = 'k'
    elif k>20 and k<40:
        test = igor[int(k-(lim/3-1))]
        colorLabel = 'g'
    else:
        test = ankita[int(k-(2*lim/3-1))]
        colorLabel = 'r'
    distances = []
    for i in range(2):
        d = 0
        for j in range(np.size(wData[0][0, ::])):
            # distance = dtwImp02(test[::, j], refSample[i][::, j])
            distance, path = fastdtw(test[::, j], refSample[i][::, j], dist=euclidean)
            d += distance
        distances.append(d/np.size(wData[0][0, ::]))
    distances = np.array(distances)
    # ic = triang(dist=distances, refPoints=refCoord)
    a0, a1, a2, a3 = icPoints(d=distances, ref=refCoord)
    ic = [a0, a1]
    print(ic)

    ax.add_artist(plt.Circle(xy=refCoord[0], radius=distances[0], alpha=0.01, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refCoord[1], radius=distances[1], alpha=0.01, edgecolor='k'))
    # ax.add_artist(plt.Circle(xy=refCoord[2], radius=distances[2], alpha=0.1, edgecolor='k'))

    plt.scatter(refCoord[0][0], refCoord[0][1], c='r')
    plt.scatter(refCoord[1][0], refCoord[1][1], c='r')
    # plt.scatter(refCoord[2][0], refCoord[2][1], c='r')

    if ic is not None:
        plt.scatter(ic[0], ic[1], marker='x', c=colorLabel)
    plt.xlim([-box, box])
    plt.ylim([-box, box])

'''
# Test:
testPoint = [5, 5]
distances = []
refCoord = np.array(refCoord)

distances.append(np.linalg.norm(refCoord[0]-testPoint))
distances.append(np.linalg.norm(refCoord[1]-testPoint))
distances.append(np.linalg.norm(refCoord[2]-testPoint))
'''

plt.show()
