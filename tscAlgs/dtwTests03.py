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

    return q1x, q1y, q2x, q2y


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
                triangCodes.append(base[vals[0]]/len(base[i]))
                triangCodes.append(base[vals[1]]/len(base[i]))
                triangCodes.append(base[vals[2]]/len(base[i]))
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
    refB = [0, d]
    return refA, refB

basis = generateOrthBasis(16) #, vals=[0, 1, 11]) # vals = [0, 1, 11] for 64, 128

coordBaseA = generateCoordinateBasis([basis[0], basis[1]])
coordBaseB = generateCoordinateBasis([basis[1], basis[2]])
coordBaseB = [[coordBaseB[0][0], coordBaseB[0][1]], [coordBaseB[1][1], coordBaseB[1][0]]]

test = []
testLabel = []
testTemplate = np.ones(16)*20
for i in range(10):
    test.append(i*testTemplate/8)
    testLabel.append('r')
testTemplate = -np.ones(16)*20
for i in range(10):
    temp = i*testTemplate/8
    temp[8] = 2*i+1
    temp[0] = i+1
    test.append(temp)
    testLabel.append('g')
'''
xaxis = np.linspace(0,1,8)
for i in range(10):
    test.append(np.sin(2*np.pi*xaxis*(i+1)/10))
    testLabel.append('g')
'''
fig, ax = plt.subplots()
plt.gca().set_aspect('equal', adjustable='box')
box = 25
plt.scatter(coordBaseA[0][0], coordBaseA[0][1], c='r', marker='x')
plt.scatter(coordBaseA[1][0], coordBaseA[1][1], c='r', marker='x')
plt.scatter(coordBaseB[0][0], coordBaseB[0][1], c='r', marker='x')
plt.scatter(coordBaseB[1][0], coordBaseB[1][1], c='r', marker='x')
plt.xlim([-box, box])
plt.ylim([-box, box])
for j in range(len(test)):
    distances = []
    for i in range(2):
        distance = dtw(test[j]*basis[i], basis[i])
        distances.append(distance)
    distances = np.array(distances)

    dummy, dummy, x, a1 = interceptPoints(d=distances, ref=[coordBaseA[0], coordBaseA[1]])

    ax.add_artist(plt.Circle(xy=coordBaseA[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=coordBaseA[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    distances = []
    for i in range(2):
        distance = dtw(test[j]*basis[i+1], basis[i+1])
        distances.append(distance)
    distances = np.array(distances)

    dummy, dummy, y, a1 = interceptPoints(d=distances, ref=[coordBaseB[1], coordBaseB[0]])

    ax.add_artist(plt.Circle(xy=coordBaseB[0], radius=distances[0], alpha=0.005, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=coordBaseB[1], radius=distances[1], alpha=0.005, edgecolor='k'))

    plt.scatter(x, y, marker='*', c=testLabel[j])

plt.show()
