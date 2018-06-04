########################################################################################################################
## cb, 01.06.18
##
## general info:
## algorithm from "Feature Subset Selection And Feature Ranking For Multivariate Time Series", Yoon et al. , 2005
##
## denoising by gauss???!!!
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
from abraxasOne.gaussFilter import gaussFilter

from abraxasOne.loadAndClean import loadAndClean

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"] #
start = np.array([400, 200, 100, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1500, 1700, 1600, 3000, 6000])
stop = np.ones([len(start)])*np.min(stop-start)+start
stop = [int(x) for x in stop]

check_data = False
if check_data:
    for i, element in enumerate(files):
        irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, 10, 2, tSample=0.0165, dirPath="")
        for j in range(len(irData)):
            plt.plot(irData[j][start[i]:stop[i],1], label=str(j))
            plt.title(files[i])
        plt.legend()
        plt.show()

dataSet = []
for i, element in enumerate(files):
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(element, 10, 2, tSample=0.0165, dirPath="")
    dummyData = np.random.rand(np.size(irData[0][start[0]:stop[0],1]))
    for j in range(len(irData)):
        irData[j] = gaussFilter(0, irData[j][start[i]:stop[i],1]) #*0 + np.random.rand(np.size(dummyData))
    #irData[1] = irData[0]
    dataSet.append(irData)

dcpc = [] # descriptive common principal component
loading = []
p = []
delta = 90
for i in range(len(dataSet)):
    X = np.array(dataSet[i]).transpose()
    corr = np.array(pd.DataFrame(X).corr())
    #plt.figure()
    #plt.title(files[i]+" Corr")
    #sns.heatmap(np.abs(corr))
    U, variance, UT = np.linalg.svd(corr, full_matrices=True)
    #variance = np.round(variance, 12)
    #U = np.round(U, 12).T
    #plt.figure()
    #plt.title(files[i]+" U")
    #sns.heatmap(np.abs(U))
    loading.append(U)
    percentVar = 100*variance/np.sum(variance)
    for j in range(len(percentVar)+1):
        if (np.sum(percentVar[0:j])>=delta):
            p.append(j)
            print(j)
            break
print(p)
p = np.max(p)
H = []
for i in range(len(dataSet)):
    L = loading[i][:p,::]
    if i==0:
        H.append(np.matmul(L.transpose(),L))
    else:
        H.append(H[i-1] + np.matmul(L.transpose(),L))
    #plt.figure()
    #plt.title(files[i]+" H")
    #sns.heatmap(np.abs(H[i])/np.max(np.max(H[i])))
H = H[len(H)-1]
V, S, VT = np.linalg.svd(H)
#V = np.round(V, 12)
DCPC = V[:p,::]
#plt.figure()
#plt.title("H")
#sns.heatmap(np.abs(H[len(dataSet)-1]))
for i in range(p):
    print(np.sum(np.abs(DCPC[i,::])*np.abs(DCPC[i,::])))
plt.figure()
plt.title("DCPC")
plt.imshow(np.abs(DCPC))
plt.colorbar()
plt.show()