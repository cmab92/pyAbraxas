import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt2
from tscAlgs.icPoints import icPoints


# distance to ref.

# distance between ref. center

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

if __name__ == '__main__':

    d = np.array([[np.sqrt(2)], [1], [1]])
    ref = np.array([[0.01, 0], [0, 1], [1, 0]])

    distances = np.array([[0.5], [np.sqrt(5)/2], [0.5]])
    refCoord = np.array([[0, 1], [1, 0], [1, 1]])

    testPoint = [1, 5]
    distances = []
    refCoord = np.array(refCoord)

    distances.append(np.linalg.norm(refCoord[0]-testPoint))
    distances.append(np.linalg.norm(refCoord[1]-testPoint))
    distances.append(np.linalg.norm(refCoord[2]-testPoint))

    ic = triang(dist=distances, refPoints=refCoord)
    print(ic)

    fig, ax = plt.subplots()

    ax.add_artist(plt.Circle(xy=refCoord[0], radius=distances[0], alpha=0.1, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refCoord[1], radius=distances[1], alpha=0.1, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=refCoord[2], radius=distances[2], alpha=0.1, edgecolor='k'))

    plt.scatter(refCoord[0][0], refCoord[0][1], c='r')
    plt.scatter(refCoord[1][0], refCoord[1][1], c='r')
    plt.scatter(refCoord[2][0], refCoord[2][1], c='r')

    plt.scatter(ic[0], ic[1], marker='x', c='g')

    box = 15
    plt.xlim([-box, box])
    plt.ylim([-box, box])

    plt.show()