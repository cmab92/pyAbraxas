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
from abraxasOne.helperFunctions import powerSetOfArray

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"]
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])
dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=True, checkData=False)#, selectSensors=selection[sel_i])

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
#print("Number of windows per dataset: ", numberOfWindows)

features = []
labels = []
trainingFeatures = []
testFeatures = []
trainingLabels = []
testLabels = []
variables = np.linspace(0,9,10)
selection = powerSetOfArray(variables)
selection = selection[1:]
scores = []
for sel_i in range(len(selection)):
    if len(selection[sel_i])>2:
        print(sel_i/len(selection)*100, "%")
        print("Selected sensors: ", selection[sel_i])

        for i in range(len(dataWindows)):
            index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
            #random.shuffle(index)
            for j in range(numberOfWindows[i]):
                f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=1, numDomFreqs=1, sampleT=0.0165, wavelet = 'haar', selectSensors=selection[sel_i])
                if j>int(2/3*numberOfWindows[i]):
                    testFeatures.append(f.T)
                    testLabels.append(i)
                else:
                    trainingFeatures.append(f.T)
                    trainingLabels.append(i)

        #shuffledTrainingLabels, shuffledTrainingFeatures = shuffleData(trainingLabels, trainingFeatures)
        #shuffledTestLabels, shuffledTestFeatures = shuffleData(testLabels, testFeatures)

        ## train svm:
        clf = []
        clf = svm.SVC(kernel='rbf')
        clf.fit(trainingFeatures, trainingLabels)
        ## test with normal data:
        prediction = []
        error = 0
        #classError = np.zeros(len(files))
        #numberOfTestsPerClass = np.zeros(len(files))
        #confMat = np.zeros([len(files), len(files)])
        for i in range(len(testLabels)):
            pred = clf.predict(testFeatures[i].reshape(1, -1))
            #confMat[int(pred),testLabels[i]] += 1
            #numberOfTestsPerClass[testLabels[i]] += 1
            if pred != testLabels[i]:
                error += 1
                #classError[testLabels[i]] += 1
        #confMat = confMat*1/numberOfTestsPerClass
        error = error/len(testLabels)*100
        print(error)
        scores.append(error)
for i in range(len(selection)):
    print("Sensors: ", selection[i], "yield: ", scores[i])