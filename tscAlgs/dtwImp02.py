import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtwImp02(v1, v2):
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


if __name__ == '__main__':
    x = [4,5,1,3]
    y = [1,2,3]
    z = [6,5,7]
    dxy = dtwImp02(v1=x, v2=y)
    dxz = dtwImp02(v1=x, v2=z)
    dyz = dtwImp02(v1=y, v2=z)
    print("D(x,y)=", dxy[-1, -1])
    print("D(x,z)=", dxz[-1, -1])
    print("D(y,z)=", dyz[-1, -1])
    print(dxy[-1, -1]+dxz[-1, -1], " = D(x,y)+D(x,z) >= D(y,z) = ", dyz[-1, -1])
