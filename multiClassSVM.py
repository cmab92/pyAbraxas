import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random

from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures
from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.helperFunctions import scaleData
from abraxasOne.helperFunctions import shuffleData
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"]
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])

fileLabels = np.array([0,1,2,3,4,1,0])

usedSensors = np.array([0,1,2,3,4,5,6,7,8,9])
print("Using following sensors: ", usedSensors)

dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=False, checkData=False, selectSensors=usedSensors)

for i in range(len(dataSet)):
    dataSet[i] = scaleData(dataSet[i], scaleAmp=True, scaleMean=False, scaleVar=True)

dataWindows = []
numberOfWindows = []
for i in range(len(dataSet)):
    windows, numOfWindows = sliceAndWindowV3(data=dataSet[i], windowWidth=50, windowShift=10, enaCheck=False, window='tukey', alpha=0.1, enaCWF=0)
    dataWindows.append(windows)
    numberOfWindows.append(numOfWindows)
dataWindows = np.array(dataWindows)
numberOfWindows = np.array(numberOfWindows)
print("Number of windows per dataset: ", numberOfWindows)

features = []
labels = []
trainingFeatures = []
testFeatures = []
trainingLabels = []
testLabels = []

for i in range(len(dataWindows)):
    index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
    #random.shuffle(index)
    for j in range(numberOfWindows[i]):
        f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=20, numDomFreqs=2, sampleT=0.0165, wavelet = 'haar')
        if j>int(2/3*numberOfWindows[i]):
            testFeatures.append(f.T)
            testLabels.append(fileLabels[i])
        else:
            trainingFeatures.append(f.T)
            trainingLabels.append(fileLabels[i])

#shuffledTrainingLabels, shuffledTrainingFeatures = shuffleData(trainingLabels, trainingFeatures)
#shuffledTestLabels, shuffledTestFeatures = shuffleData(testLabels, testFeatures)

## train svm:
clf = svm.SVC(kernel='rbf')
clf.fit(trainingFeatures, trainingLabels)
## test with normal data:
prediction = []
error = 0
classError = np.zeros(len(files)-2)
numberOfTestsPerClass = np.zeros(len(files)-2)
confMat = np.zeros([len(files)-2, len(files)-2])
for i in range(len(testLabels)):
    pred = clf.predict(testFeatures[i].reshape(1, -1))
    confMat[int(pred),testLabels[i]] += 1
    numberOfTestsPerClass[testLabels[i]] += 1
    if pred != testLabels[i]:
        error += 1
        classError[testLabels[i]] += 1
confMat = confMat*1/numberOfTestsPerClass
error = error/len(testLabels)*100
print("Class. Error")
print(error)
print("per class:")
print(classError/numberOfTestsPerClass*100)
plotMatrixWithValues(confMat)
