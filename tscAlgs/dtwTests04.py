import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw(v1, v2):
    n = len(v1)
    m = len(v2)
    S = np.zeros([n+1, m+1])
    for i in range(m):
        S[0, i+1] = np.inf
    for i in range(n):
        S[i+1, 0] = np.inf
    for i in range(n):
        for j in range(m):
            # cost = np.linalg.norm(x=(v1[i] - v2[j]), ord=2)
            cost = np.sqrt(np.square(v1[i] - v2[j]))
            S[i+1, j+1] = cost + np.min([S[i, j+1], S[i+1, j], S[i, j]])
    return S[-1, -1]


def interceptPoints(d, ref):

    Ax = ref[0][0]  # x-coord. circle 1
    Bx = ref[1][0]  # x-coord. circle 2
    Ay = ref[0][1]  # y-coord. circle 1
    By = ref[1][1]  # y-coord. circle 2
    a = d[0]        # radius circle 1
    b = d[1]        # radius circle 2

    c = np.sqrt(np.square(Bx - Ax) + np.square(By - Ay))

    x = (np.square(a) + np.square(c) - np.square(b))/(2*c)
    y = np.sqrt(np.square(a) - np.square(x))

    q1x = Ax + x*(Bx - Ax)/c - y*(By - Ay)/c
    #q2x = Ax + x*(Bx - Ax)/c + y*(By - Ay)/c
    q1y = Ay + x*(By - Ay)/c + y*(Bx - Ax)/c
    #q2y = Ay + x*(By - Ay)/c - y*(Bx - Ax)/c

    return q1x, q1y  #, q2x, q2y


def generateOrthBasis(length, vals=None):
    base = np.array([[1, 1], [1, -1]])
    l = np.log2(length) -1
    if l.is_integer():
        for i in range(int(l)):
            current = []
            for j in range(len(base)):
                current.append(np.reshape([base[j], base[j]], np.size([base[j], base[j]])))
                current.append(np.reshape([base[j], -1*base[j]], np.size([base[j], base[j]])))
            base = current
        orthogonality = True
        for i in range(len(base)):
            for j in range(len(base)):
                if np.dot(base[i], base[j])!=0 and i!=j:
                    orthogonality = False
        if orthogonality:
            if vals is None:
                for i in range(len(base)):
                    for j in range(len(base)):
                        d01, dummy = fastdtw(base[i], base[j], dist=euclidean)
                        for k in range(len(base)):
                            d20, dummy = fastdtw(base[k], base[i], dist=euclidean)
                            d21, dummy = fastdtw(base[k], base[j], dist=euclidean)
                            if d01==d20 and d20==d21 and d01!=0:
                                triangCodes = []
                                triangCodes.append(base[i]/len(base[i]))
                                triangCodes.append(base[j]/len(base[i]))
                                triangCodes.append(base[k]/len(base[i]))
                                base = triangCodes
                                print("Vals are (i, j, k):=", i, j, k)
                                return base
            elif isinstance(vals, (np.ndarray, list)):
                triangCodes = []
                triangCodes.append(base[vals[0]]/len(base[0]))
                triangCodes.append(base[vals[1]]/len(base[1]))
                triangCodes.append(base[vals[2]]/len(base[2]))
                base = triangCodes
                return base

        else:
            print("(generateOrthBasis) Failed")
            quit()
    else:
        print("(generateOrthBasis) Wrong length")
        quit()


def generateCoordinateBasis(orthBasis):
    d, dummy = fastdtw(orthBasis[0], orthBasis[1], dist=euclidean)
    # d = dtw(orthBasis[0], orthBasis[1])
    refA = [0, 0]
    refB = [0, 1]
    return refA, refB


def getRabeData(sensors, length, shift=256):
    from abraxas4.abraxasFrame import AbraxasFrame
    b = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=length, windowShift=shift, numFreqs=0, numCoeffs=0,
                          enaStatFeats=False, featNormMethod='none', trainFraction=2/3, waveletLvl1=False,
                          randomSortTT=False, classSortTT=True)

    b.setWindowFunction(functionName='rect', alpha=0.25)
    # b.plotWindowFunction()

    b.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
    b.selectSensorSubset(selectedSensors=[], sensorType='fr')
    b.selectSensorSubset(selectedSensors=sensors, sensorType='ir')
    #b.selectSensorSubset(selectedSensors=[2], sensorType='ir')

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
            igor.append(wData[i]*1)
        if wLabels[i]==1:
            ankita.append(wData[i]*1)
        if wLabels[i]==2:
            markus.append(wData[i]*1)
    return igor, ankita, markus


'''
---
'''

length = 512
igor, ankita, markus = getRabeData(sensors=[0, 1, 4, 8], length=length, shift=256)

basis = generateOrthBasis(length, vals=[0, 1, 11]) #, vals=[0, 1, 11]) # vals = [0, 1, 11] for 64, 128, 256

coordBaseA = generateCoordinateBasis([basis[0], basis[1]])
coordBaseB = generateCoordinateBasis([basis[1], basis[2]])
coordBaseB = [[coordBaseB[0][0], coordBaseB[0][1]], [coordBaseB[1][1], coordBaseB[1][0]]]

test = []
testLabel = []
testMarker = []

factor = 1
# plt.figure()
for i in range(len(igor)-1):
    test.append(igor[i+1]*factor)
    # plt.plot(igor[i]*factor, 'g')
    testLabel.append('g')
    testMarker.append('x')
# plt.figure()
for i in range(len(ankita)-1):
    test.append(ankita[i+1]*factor)
    # plt.plot(ankita[i]*factor, 'b')
    testLabel.append('b')
    testMarker.append('o')
# plt.figure()
for i in range(len(markus)-1):
    test.append(markus[i+1]*factor)
    # plt.plot(markus[i]*factor, 'r')
    testLabel.append('r')
    testMarker.append('*')

fig, ax = plt.subplots()
#plt.gca().set_aspect('equal', adjustable='box')
box = 25
plt.scatter(coordBaseA[0][0], coordBaseA[0][1], c='r', marker='x')
plt.scatter(coordBaseA[1][0], coordBaseA[1][1], c='r', marker='x')
plt.scatter(coordBaseB[0][0], coordBaseB[0][1], c='r', marker='x')
plt.scatter(coordBaseB[1][0], coordBaseB[1][1], c='r', marker='x')
# plt.xlim([-box, box])
# plt.ylim([-box, box])

for i in range(len(test)):
    #print(i/len(test))
    distances = []
    for j in range(2):
        d = 0
        for k in range(np.size(test[i][0, ::])):
            distance, dummy = fastdtw(test[i][::, k], basis[j], dist=euclidean)
            d += distance
        distances.append(distance)
    distances = np.array(distances)

    x1, y1 = interceptPoints(d=distances, ref=[coordBaseA[0], coordBaseA[1]])
    # plt.scatter(x1, y1, marker='*', c=testLabel[i])

    ax.add_artist(plt.Circle(xy=coordBaseA[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=coordBaseA[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    distances = []
    for j in range(2):
        d = 0
        for k in range(np.size(test[i][0, ::])):
            distance, dummy = fastdtw(test[i][::, k], basis[j+1], dist=euclidean)
            d += distance
        distances.append(distance)
    distances = np.array(distances)

    x2, y2 = interceptPoints(d=distances, ref=[coordBaseB[1], coordBaseB[0]])
    # plt.scatter(x2, y2, marker='o', c=testLabel[i])

    ax.add_artist(plt.Circle(xy=coordBaseB[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=coordBaseB[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    print(y1, y2)
    plt.scatter(y1, y2, marker=testMarker[i], c=testLabel[i])

plt.show()
