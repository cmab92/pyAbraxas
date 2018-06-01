########################################################################################################################
## cb, 01.06.18
##
## general info:
## algorithm from "Feature Subset Selection And Feature Ranking For Multivariate Time Series", Yoon et al. , 2005
##
## function inputs:
##
## functions output:
##
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abraxasOne.loadAndClean import loadAndClean

files = ["ankita.txt", "chris_asymm.txt", "chris_c.txt", "chris_pos2.txt", "igor.txt", "igor2.txt", "ankita_pos2_lrRl.txt"]
start = np.array([250, 200, 100, 100, 100, 800, 100])
stop = np.array([1750, 1400, 1500, 1700, 3500, 6400, 2700])
stop = np.ones([len(start)])*np.min(stop-start)+start
stop = [int(x) for x in stop]

dataSet = []
for i, element in enumerate(files):
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, 10, 2, tSample=0.0165, dirPath="")
    for j in range(len(irData)):
        irData[j] = irData[j][start[i]:stop[i],1]
    dataSet.append(irData)

dcpc = [] # descriptive common principal component
loading = []
p = []
delta = 90
for i in range(len(dataSet)):
    X = dataSet[i]
    corr = pd.DataFrame(X).corr()
    #sns.heatmap(corr)
    U, S, UT = np.linalg.svd(corr)
    loading.append(U)
    variance = S
    percentVar = 100*variance/np.sum(variance)
    print(percentVar)
    for i in range(len(percentVar)):
        if (np.sum(percentVar[0:i])>delta):
            print(i)
            print("test")
            break
    print("test2")
    p.append(np.sum(percentVar[percentVar>90]/percentVar[percentVar>90]))

plt.show()


    #U, S, UT = np.linalg.svd(, full_matrices=True)