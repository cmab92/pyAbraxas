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
import seaborn as sns
from decimal import localcontext
from abraxasOne.gaussFilter import gaussFilter
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues
from abraxasOne.loadAndClean import loadAndClean
from CLeVerOne import computeDCPC
from CLeVerOne import rankVariables
from CLeVerOne import clusterVariables

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"] #
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])
stop = np.ones([len(start)])*np.min(stop-start)+start
stop = [int(x) for x in stop]

numOfVariables = 10

check_data = False
filter_data = True

if check_data:
    for i, element in enumerate(files):
        irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, 10, 2, tSample=0.0165, dirPath="", normAnalog=False)
        for j in range(len(irData)):
            plt.plot(irData[j][start[i]:stop[i],1], label=str(j))
            plt.title(files[i])
        plt.legend()
        plt.show()

dataSet = []
for i, element in enumerate(files):
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, 10, 2, tSample=0.0165, dirPath="", normAnalog=False)
    dummyData = np.random.rand(np.size(irData[0][start[0]:stop[0],1]))
    if filter_data:
        for j in range(len(irData)):
            irData[j] = gaussFilter(0, irData[j][start[i]:stop[i],1], AMP=1, MEAN=0, SIGMA=0.15)
    else:
        for j in range(len(irData)):
            irData[j] = irData[j][start[i]:stop[i], 1]
    factor = 0.01
    irData[4] = irData[1] + np.random.rand(np.size(irData[0]))*factor
    irData[7] = irData[1] + np.random.rand(np.size(irData[0]))*factor
    irData[5] = irData[1] + np.random.rand(np.size(irData[0]))*factor
    irData[3] = irData[1] + np.random.rand(np.size(irData[0]))*factor
    # normalize:
    for j in range(len(irData)):
        irData[j] = (irData[j]-np.min(irData[j]))/(np.max(irData[j]-np.min(irData[j])))
        irData[j] = (irData[j] - np.mean(irData[j])) / np.sqrt(np.var(irData[j]))
    irData = np.array(irData).T
    dataSet.append(irData)


#index = rankVariables(dataSet=dataSet, delta=0.8, k=6)
selected = clusterVariables(dataSet=dataSet, delta=90, k=4)
print(selected)
#print(index)
    # plot selected:
if 0==1:
    for i in range(len(dataSet)):
        plt.figure()
        for j in range(numOfVariables):
            if j in index:
                label_ = "selected "+str(j)
            else:
                label_ = "NOT selected "+str(j)
            plt.plot(dataSet[i][j], label=label_)
            plt.title(files[i])
        plt.legend()
if 0==1:
    for i in range(len(dataSet)):
        plt.figure()
        for j in range(numOfVariables):
            if j not in index:
                label_ = str(j)+", dataSet "+str(files[i])
                plt.plot(dataSet[i][j], label=label_)
        plt.legend()
        plt.title("all not selected"+files[i])

plt.show()