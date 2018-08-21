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
    q2x = Ax + x*(Bx - Ax)/c + y*(By - Ay)/c
    q1y = Ay + x*(By - Ay)/c + y*(Bx - Ax)/c
    q2y = Ay + x*(By - Ay)/c - y*(Bx - Ax)/c

    return np.round(q1x, 10), np.round(q1y, 10)#, np.round(q2x, 10), np.round(q2y, 10)  #, q2x, q2y


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

def triang(dist, refPoints):

    icx00, icy00, icx10, icy10 = interceptPoints(d=np.array([[dist[0]], [dist[1]]]), ref=np.array([refPoints[0],
                                                                                            refPoints[1]]))

    icx01, icy01, icx11, icy11 = interceptPoints(d=np.array([[dist[1]], [dist[2]]]), ref=np.array([refPoints[1],
                                                                                            refPoints[2]]))

    icx02, icy02, icx12, icy12 = interceptPoints(d=np.array([[dist[2]], [dist[0]]]), ref=np.array([refPoints[2],
                                                                                            refPoints[0]]))

    s0 = np.reshape([icx00, icy00], 2), np.reshape([icx10, icy10], 2)
    s1 = np.reshape([icx01, icy01], 2), np.reshape([icx11, icy11], 2)
    s2 = np.reshape([icx02, icy02], 2), np.reshape([icx12, icy12], 2)

    if np.round(np.linalg.norm(s0[0]-s1[0]), 10)==0 or np.round(np.linalg.norm(s0[0]-s1[1]), 10)==0:
        if np.round(np.linalg.norm(s0[0]-s2[0]), 10)==0 or np.round(np.linalg.norm(s0[0]-s2[1]), 10)==0:
            return s0[0]
    if np.round(np.linalg.norm(s0[1]-s1[0]), 10)==0 or np.round(np.linalg.norm(s0[1]-s1[1]), 10)==0:
        if np.round(np.linalg.norm(s0[1]-s2[0]), 10)==0 or np.round(np.linalg.norm(s0[1]-s2[1]), 10)==0:
            return s0[1]



'''
---
'''

length = 512
igor, ankita, markus = getRabeData(sensors=[0], length=length, shift=128)

base = generateOrthBasis(length) #, vals=[0, 1, 11]) # vals = [0, 1, 11] for 64, 128, 256
basis = []
xaxis = np.linspace(0, 1, length)
# basis.append(np.sin(2*np.pi*xaxis*0)/length)
# basis.append(np.sin(2*np.pi*xaxis*2)/length)
basis.append(base[1]/length)
basis.append(base[11]/length)
basis.append(base[19]/length)
# basis.append(base[91]/length)

basis[0][basis[0]==-1] = 0
basis[1][basis[1]==-1] = 0

print("ONB / all zero?")
print(np.dot(basis[0], basis[1]))
print(np.dot(basis[1], basis[2]))
print(np.dot(basis[0], basis[2]))

d0=fastdtw(basis[0], basis[1], dist=euclidean)[0]
d1=fastdtw(basis[1], basis[2], dist=euclidean)[0]
d2=fastdtw(basis[0], basis[2], dist=euclidean)[0]
print("DTW Costs (Basis): ", d0, d1, d2)

coordBaseA = generateCoordinateBasis([basis[0], basis[1]])
coordBaseB = generateCoordinateBasis([basis[1], basis[2]])

test = []
testLabel = []
testMarker = []

# '''
factor = 1
for i in range(len(igor)-1):
    test.append(igor[i+1]*factor)
    testLabel.append('g')
    testMarker.append('x')
for i in range(len(ankita)-1):
    test.append(ankita[i+1]*factor)
    testLabel.append('b')
    testMarker.append('o')
for i in range(len(markus)-1):
    test.append(markus[i+1]*factor)
    testLabel.append('r')
    testMarker.append('*')
# '''

'''
xaxis = np.linspace(0, 1, length)
for i in range(10):
    test.append([np.sin(2*np.pi*xaxis*i*10)])
    testLabel.append('k')
    testMarker.append('H')
for i in range(10):
    test.append([np.sin(2*np.pi*xaxis*i*10) + 1])
    plt.plot(np.sin(2*np.pi*xaxis*i*10)+1)
    testLabel.append((0.5,0.5,1))
    testMarker.append('H')
plt.show()
'''

'''
test.append(igor[0])
testLabel.append('r')
testMarker.append('o')
test.append(ankita[0])
testLabel.append('g')
testMarker.append('H')
test.append(markus[0])
testLabel.append('b')
testMarker.append('x')

print(fastdtw(ankita[0], igor[0], dist=euclidean)[0])
print(fastdtw(ankita[0], markus[0], dist=euclidean)[0])
print(fastdtw(markus[0], igor[0], dist=euclidean)[0])
'''


fig, ax = plt.subplots()
box = 25

for i in range(len(test)):
    distances = []
    for j in range(2):
        d = 0
        try:
            temp = test[i][0, ::]
            for k in range(np.size(temp)):
                dist, dummy = fastdtw(test[i][::, k], basis[j], dist=euclidean)
                d += dist
            distances.append(d/np.size(temp))
        except TypeError:
            dist, dummy = fastdtw(test[i], basis[j], dist=euclidean)
            d += dist
            distances.append(d)
    distances = np.array(distances)

    x1, y1 = interceptPoints(d=distances, ref=[coordBaseA[0], coordBaseA[1]])
    # plt.scatter(x1, y1, marker='*', c=testLabel[i])

    # ax.add_artist(plt.Circle(xy=coordBaseA[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    # ax.add_artist(plt.Circle(xy=coordBaseA[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    distances = []
    for j in range(2):
        d = 0
        try:
            temp = test[i][0, ::]
            for k in range(np.size(temp)):
                dist, dummy = fastdtw(test[i][::, k], basis[j+1], dist=euclidean)
                d += dist
            distances.append(d/np.size(temp))
        except TypeError:
            dist, dummy = fastdtw(test[i], basis[j+1], dist=euclidean)
            d += dist
            distances.append(d)

    distances = np.array(distances)

    x2, y2 = interceptPoints(d=distances, ref=[coordBaseB[0], coordBaseB[1]])

    # ax.add_artist(plt.Circle(xy=coordBaseB[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    # ax.add_artist(plt.Circle(xy=coordBaseB[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    print(x1, y1, x2, y2)
    '''
    for j in range(2):
        try:
            temp = test[i][0, ::]
            for k in range(np.size(temp)):
                plt.figure(1)
                plt.plot(test[i][::, k]*basis[j+2], 'r')
                plt.plot(test[i][::, k]*basis[j], 'g')
                plt.plot(basis[j+2], 'r--')
                plt.plot(basis[j], 'g--')
                plt.figure(2)
                plt.plot(test[i][::, k], 'b')
        except TypeError:
            plt.plot(test[i]*basis[j+2], 'r')
            plt.plot(test[i], 'b')
    plt.show()
    '''
    plt.scatter(x1, x2, marker='*', c=testLabel[i])
print("...")
plt.show()
