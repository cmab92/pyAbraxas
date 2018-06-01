import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## load data
from abraxasOne.loadAndClean import loadAndClean
files = ["ankita.txt", "chris_asymm.txt", "chris_c.txt", "chris_pos2.txt", "igor.txt", "igor2.txt", "ankita_pos2_lrRl.txt"]

def autoCorr(x, lagUpTo=1):
    y = []
    for i in range(lagUpTo):
        y.append(pd.Series.autocorr(pd.Series(x), lag=i))
    return np.array(y)

corrTotal = []
for i, element in enumerate(files):
    dir = element
    irData, forceData, quatData, linAccData, angVecData = loadAndClean(dir, 10, 2, tSample = 0.0165, dirPath = "")

    X = []
    autoCorrX = []
    corrX = []
    for i in range(len(irData)):
        data = np.abs(irData[i][::,1])
        X.append(data)
        autoCorrX.append(autoCorr(x = data, lagUpTo=200))
        corrXtemp = []
        for j in range(len(irData)):
            corrXtemp.append(pd.Series(data).corr(other=pd.Series(irData[j][::,1])))
        corrX.append(corrXtemp)
        plt.plot(autoCorr(x=data, lagUpTo=200))
    corrX = np.array(corrX)
    plt.title(dir)
    corrTotal.append(corrX)
    plt.show()

corrTotal = np.array(corrTotal)
corrTotal = np.sum(corrTotal,0)/len(corrTotal)
#fig, ax = plt.subplots()
#min_val, max_val = 0, 10
#intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
#ax.matshow(corrTotal, cmap=plt.cm.Blues)
#for i in range(len(irData)):
#    for j in range(len(irData)):
#        c = round(corrTotal[j,i], 2)
#        ax.text(i, j, str(c), va='center', ha='center')
#plt.show()
