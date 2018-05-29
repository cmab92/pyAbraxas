import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as skd
import pandas as pd

## load data
from abraxasOne.loadAndClean import loadAndClean
irData, forceData, quatData, linAccData, angVecData = loadAndClean("_20185161226.txt", 10, 4, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
#irData, forceData, quatData, linAccData, angVecData = loadAndClean("langsamChris_20185291452.txt", 10, 4, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
#irData, forceData, quatData, linAccData, angVecData = loadAndClean("chris_20185291448.txt", 10, 4, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
#irData, forceData, schnellChris_20185291452quatData, linAccData, angVecData = loadAndClean("sashi_20185291449.txt", 10, 4, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
#dir = "toshiba2_20185291523.txt"
#irData, forceData, quatData, linAccData, angVecData = loadAndClean(dir, 10, 2, tSample = 0.0165, dirPath = "/home/bonenberger/pyAbraxas/")
## denoise
from abraxasOne.gaussFilter import gaussFilter

def autoCorr(x, lagUpTo=1):
    y = []
    for i in range(lagUpTo):
        y.append(pd.Series.autocorr(pd.Series(x), lag=i))
    return np.array(y)

X = []
autoCorrX = []
corrX = []
for i in range(len(irData)):
    data = irData[i][::,1]
    #plt.plot(data)
    #plt.show()
    #data = 1/np.max(data)/(1-np.min(data)/np.max(data))*data - (1/np.max(data)/(1-np.min(data)/np.max(data))*np.min(data))*np.ones([len(data)])
    X.append(data)
    autoCorrX.append(autoCorr(x = data, lagUpTo=200))
    corrXtemp = []
    for j in range(len(irData)):
        corrXtemp.append(pd.Series(data).corr(other=pd.Series(irData[j][::,1])))
    corrX.append(corrXtemp)
corrX = np.array(corrX)
print(corrX)

fig, ax = plt.subplots()

min_val, max_val = 0, 10

intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

ax.matshow(corrX, cmap=plt.cm.Blues)

for i in range(len(irData)):
    for j in range(len(irData)):
        c = round(corrX[j,i], 2)
        ax.text(i, j, str(c), va='center', ha='center')
plt.title(dir)
plt.show()