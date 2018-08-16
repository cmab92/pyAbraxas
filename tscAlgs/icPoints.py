import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt2


# distance to ref.

# distance between ref. center

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

if __name__ == '__main__':

    d = np.array([[np.sqrt(2)], [1]])
    ref = np.array([[1, 0], [0, 1]])

    icx0, icy0, icx1, icy1 = icPoints(d, ref)

    fig, ax = plt.subplots()

    ax.add_artist(plt.Circle(xy=ref[0], radius=d[0], alpha=0.25, edgecolor='k'))
    ax.add_artist(plt.Circle(xy=ref[1], radius=d[1], alpha=0.25, edgecolor='k'))

    plt.scatter(ref[0][0], ref[0][1], c='r')
    plt.scatter(ref[1][0], ref[1][1], c='r')

    plt.scatter(icx0, icy0, marker='x', c='g')
    plt.scatter(icx1, icy1, marker='x', c='g')

    plt.ylim([-2, 2])
    plt.xlim([-2, 2])

    plt.show()