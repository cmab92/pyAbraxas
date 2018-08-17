import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtwImp01(v1, v2):
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
    return S[-1, -1]

if __name__ == '__main__':
    v1 = [1,2,3,4,5]
    v2 = [1,1,1,1,1]
    dtw = dtwImp01(v1=v1, v2=v2)
    print(dtw)
    d, path = fastdtw(v1, v2, dist=euclidean)
    print(d)