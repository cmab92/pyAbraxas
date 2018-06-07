########################################################################################################################
## cb, 01.06.18
##
## general info:
## algorithm from "Feature Subset Selection And Feature Ranking For Multivariate Time Series", Yoon et al. , 2005
##
## denoising by gauss???!!!!
##
## function inputs:
##
## functions output:
##
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from abraxasOne.gaussFilter import gaussFilter
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues
from scipy.cluster.vq import kmeans, vq

def computeDCPC(dataSet, delta):
    loading = []
    p = []
    for i in range(len(dataSet)):
        X = np.array(dataSet[i])
        if np.linalg.cond(X)>10**9:
            print("Ill-conditioned matrix X... Cond. number is: \n", np.linalg.cond(X))
        ########################################################################
        #### U, S, VT = svd(X)
        #### W, D, WT = svd(C), where C is the correlation matrix 1/n*X'*X
        ### W = VT' except sign ambiguity
        ### D = 1/n*S²
        ########################################################################
        #C = np.matmul(X.T,X)/len(X)
        #W, D, WT = np.linalg.svd(C)
        #D = np.diag(D)
        U, S, VT = np.linalg.svd(X)
        if np.linalg.cond(VT)>10**9:
            print("Ill-conditioned matrix VT... Cond. number is: \n", np.linalg.cond(X))
        S = 1/len(X)*np.diag(S)**2
        variance = np.array(sorted(np.diag(S), reverse=True))
        loading.append(VT.T)
        percentVar = 100*variance/np.sum(variance)
        for j in range(len(percentVar)+1):
            if (np.sum(percentVar[0:j])>=delta):
                p.append(j)
                break
    p = np.max(p)
    H = []
    for i in range(len(dataSet)):
        L = loading[i][:p,::]
        if i==0:
            H.append(np.matmul(L.transpose(),L))
        else:
            H.append(H[i-1] + np.matmul(L.transpose(),L))
    H = H[len(H)-1]
    V, S, VT = np.linalg.svd(H)
    DCPC = V[:p,::]  # descriptive common principal component
    return DCPC

def rankVariables(dataSet, delta, k):
    DCPC = computeDCPC(dataSet=dataSet, delta=delta)
    score = []
    for i in range(np.size(DCPC[0,::])):
        score.append(np.linalg.norm(DCPC[::,i], ord=2))
    score = np.array(score)
    index = score.argsort()[-k:][::-1]
    return index

def clusterVariables(dataSet, delta, k):
    DCPC = np.array(computeDCPC(dataSet=dataSet, delta=delta)).T
    centroid,_ = kmeans(DCPC, k)
    index,_ = vq(DCPC,centroid)
    selected = []
    for i in range(k):
        distance = []
        for j in range(len(DCPC)):
            distance.append( np.linalg.norm(centroid[i]-DCPC[j,::], ord=2 ) )
        selected.append(np.argmin(distance))
    plotMatrixWithValues(DCPC, show=False)
    plotMatrixWithValues(centroid, show=False)
    return selected