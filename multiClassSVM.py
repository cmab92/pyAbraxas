import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures
from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.helperFunctions import scaleData

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
         "igor2.txt"]
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])
dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=False, checkData=False)

for i in range(len(dataSet)):
    dataSet[i] = scaleData(dataSet[i], scaleAmp=True, scaleMean=True, scaleVar=True)

dataWindows = []
numberOfWindows = []
for i in range(len(dataSet)):
    windows, numOfWindows = sliceAndWindowV3(data=dataSet[i], windowWidth=3, sampleT=0.0165, enaCheck=False, window='tukey', alpha=0.05, enaCWF=0)
    #print(numOfWindows)
    #print(windows)
    #for j in range(numOfWindows):
    #    plt.figure()
    #    plt.plot(windows[j])
    #plt.show()
    dataWindows.append(windows)
    numberOfWindows.append(numOfWindows)
dataWindows = np.array(dataWindows)
numberOfWindows = np.array(numberOfWindows)
print("number of windows: ", numberOfWindows)

features = []
labels = []
trainingFeatures = []
testFeatures = []
trainingLabels = []
testLabels = []

for i in range(len(dataWindows)):
    plt.figure()
    for j in range(numberOfWindows[i]):
        f = extractSpectralFeatures(dataWindow=dataWindows[i][j], numDomCoeffs=10, numDomFreqs=10, sampleT=0.0165, wavelet = 'db16')
        if j>int(1/2*numberOfWindows[i]):
            testFeatures.append(f.T)
            testLabels.append(i)
        else:
            trainingFeatures.append(f.T)
            trainingLabels.append(i)
        #plt.plot(f)
    #plt.show()

## train svm:
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(trainingFeatures, trainingLabels)
## test with normal data:
prediction = []
error = 0
for i in range(np.size(testLabels)):
    pred = clf.predict(testFeatures[i].reshape(1, -1))
    if pred!=testLabels[i]:
        error+=1
error = error/np.size(testLabels)*100
print(error)