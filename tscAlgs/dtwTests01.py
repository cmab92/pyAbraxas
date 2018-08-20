import numpy as np
import matplotlib.pyplot as plt

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
            cost = np.linalg.norm(v1[i] - v2[j])
            S[i+1, j+1] = cost + np.min([S[i, j+1], S[i+1, j], S[i, j]])
    # u,v,w = np.linalg.svd(S[1::, 1::])
    # S = S[1:, 1:]
    # eigVal = np.linalg.eigvals(S)
    # S = np.sum(np.abs(eigVal))/len(eigVal)
    # eigVec = np.linalg.eig(S)
    # S = np.linalg.det(w)

    return S[-1, -1]

def icPoints(d, ref):

    if ref[0][0]>ref[1][0]:
        Ax = ref[1][0]
        Bx = ref[0][0]
        Ay = ref[1][1]
        By = ref[0][1]
        d0 = d[1]
        d1 = d[0]
    else:
        Ax = ref[0][0]
        Bx = ref[1][0]
        Ay = ref[0][1]
        By = ref[1][1]
        d0 = d[0]
        d1 = d[1]

    c = np.sqrt((Bx - Ax)**2 + (By - Ay)**2)

    sx0 = (d0**2 + c**2 - d1**2)/(2*c)
    sy0 = np.sqrt(d0**2 - sx0**2)

    icx0 = Ax + sx0*(Bx - Ax)/c - sy0*(By - Ay)/c
    icy0 = Ay + sx0*(By - Ay)/c + sy0*(Bx - Ax)/c
    icx1 = Ax + sx0*(Bx - Ax)/c + sy0*(By - Ay)/c
    icy1 = Ay + sx0*(By - Ay)/c - sy0*(Bx - Ax)/c

    return icx0, icy0, icx1, icy1

def triang(dist, refPoints):

    icx00, icy00, icx10, icy10 = icPoints(d=np.array([[dist[0]], [dist[1]]]), ref=np.array([refPoints[0],
                                                                                            refPoints[1]]))

    icx01, icy01, icx11, icy11 = icPoints(d=np.array([[dist[1]], [dist[2]]]), ref=np.array([refPoints[1],
                                                                                            refPoints[2]]))

    icx02, icy02, icx12, icy12 = icPoints(d=np.array([[dist[2]], [dist[0]]]), ref=np.array([refPoints[2],
                                                                                            refPoints[0]]))
    '''
    fig, ax = plt.subplots()

    ax.add_artist(plt.Circle(xy=refPoints[0], radius=dist[0], alpha=0.25, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refPoints[1], radius=dist[1], alpha=0.25, edgecolor='k'))

    plt.scatter(refPoints[0][0], refPoints[0][1], c='r')
    plt.scatter(refPoints[1][0], refPoints[1][1], c='r')

    plt.scatter(icx00, icy00, marker='x', c='g')
    plt.scatter(icx10, icy10, marker='x', c='g')

    ax.add_artist(plt.Circle(xy=refPoints[1], radius=dist[1], alpha=0.25, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refPoints[2], radius=dist[2], alpha=0.25, edgecolor='k'))

    plt.scatter(refPoints[1][0], refPoints[1][1], c='r')
    plt.scatter(refPoints[2][0], refPoints[2][1], c='r')

    plt.scatter(icx01, icy01, marker='x', c='g')
    plt.scatter(icx11, icy11, marker='x', c='g')

    ax.add_artist(plt.Circle(xy=refPoints[2], radius=dist[2], alpha=0.25, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refPoints[0], radius=dist[0], alpha=0.25, edgecolor='k'))

    plt.scatter(refPoints[2][0], refPoints[2][1], c='r')
    plt.scatter(refPoints[0][0], refPoints[0][1], c='r')

    plt.scatter(icx02, icy02, marker='x', c='g')
    plt.scatter(icx12, icy12, marker='x', c='g')

    box = 60
    plt.ylim([-box, box])
    plt.xlim([-box, box])
    plt.show()
    '''

    s0 = np.reshape([icx00, icy00], 2), np.reshape([icx10, icy10], 2)
    s1 = np.reshape([icx01, icy01], 2), np.reshape([icx11, icy11], 2)
    s2 = np.reshape([icx02, icy02], 2), np.reshape([icx12, icy12], 2)

    if np.round(np.linalg.norm(s0[0]-s1[0]), 10)==0 or np.round(np.linalg.norm(s0[0]-s1[1]), 10)==0:
        if np.round(np.linalg.norm(s0[0]-s2[0]), 10)==0 or np.round(np.linalg.norm(s0[0]-s2[1]), 10)==0:
            return s0[0]
    if np.round(np.linalg.norm(s0[1]-s1[0]), 10)==0 or np.round(np.linalg.norm(s0[1]-s1[1]), 10)==0:
        if np.round(np.linalg.norm(s0[1]-s2[0]), 10)==0 or np.round(np.linalg.norm(s0[1]-s2[1]), 10)==0:
            return s0[1]

symbol1 = 1
symbol2 = -1

sample0 = np.array([symbol1, symbol1, symbol1, symbol1, symbol1, symbol1, symbol1, symbol1])
sample1 = np.array([symbol1, symbol1, symbol2, symbol2, symbol1, symbol1, symbol2, symbol2])
sample2 = np.array([symbol1, symbol2, symbol1, symbol2, symbol1, symbol2, symbol1, symbol2])
test = []
testLabel = []
test.append(np.array([symbol1, symbol2, symbol2, symbol1, symbol1, symbol2, symbol2, symbol1]))
testLabel.append('r')
test.append(np.array([symbol1, symbol1, symbol1, symbol1, symbol2, symbol2, symbol2, symbol2]))
testLabel.append('g')
test.append(np.array([symbol1, symbol1, symbol2, symbol2, symbol2, symbol2, symbol1, symbol1]))
testLabel.append('b')
test.append(np.array([symbol1, symbol2, symbol1, symbol2, symbol2, symbol1, symbol2, symbol1]))
testLabel.append('k')
test.append(np.array([symbol1, symbol2, symbol2, symbol1, symbol2, symbol1, symbol1, symbol2]))
testLabel.append('y')

refSample = [sample0, sample1, sample2]

d01 = dtw(sample1, sample0)
d20 = dtw(sample2, sample0)
d21 = dtw(sample2, sample1)

ref0 = [0, 0]
ref1 = [0, d01]
ref2x, ref2y, dummy, dummy = icPoints(d=np.array([d20, d21]), ref=np.array([ref0, ref1]))
ref2 = [ref2x, ref2y]

refCoord = [ref0, ref1, ref2]

fig, ax = plt.subplots()
box = 150
#ax.add_artist(plt.Circle(xy=refCoord[0], radius=d20, alpha=0.1, edgecolor='k'))
#ax.add_artist(plt.Circle(xy=refCoord[1], radius=d21, alpha=0.1, edgecolor='k'))

plt.scatter(refCoord[0][0], refCoord[0][1], c='r', marker='x')
plt.scatter(refCoord[1][0], refCoord[1][1], c='r', marker='x')
plt.scatter(refCoord[2][0], refCoord[2][1], c='r', marker='x')
plt.xlim([-box, box])
plt.ylim([-box, box])

for j in range(len(test)):
    distances = []
    for i in range(3):
        distance = dtw(test[j]*refSample[i], refSample[i])
        distances.append(distance)
    distances = np.array(distances)
    # ic = triang(dist=distances, refPoints=refCoord)
    ic = None
    a0, a1, a2, a3 = icPoints(d=distances, ref=refCoord)
    ic0 = [a0, a1]
    ic1 = [a2, a3]
    print(ic0)
    print(ic1)

    ax.add_artist(plt.Circle(xy=refCoord[0], radius=distances[0], alpha=0.01, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refCoord[1], radius=distances[1], alpha=0.01, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refCoord[2], radius=distances[2], alpha=0.1, edgecolor='k'))

    if ic is not None:
        plt.scatter(ic[0], ic[1], marker='x', c='g')

    plt.scatter(ic0[0], ic0[1], marker='o', c=testLabel[j])
    plt.xlim([-box, box])
    plt.ylim([-box, box])


plt.show()