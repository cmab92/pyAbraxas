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
import serial
from six.moves import cPickle

#files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt", "chris1.txt", "stefan.txt", "ben.txt", "markus.txt", "markusSchnell.txt"]
#start = np.array([600, 300, 50, 100, 100, 100, 3500, 500, 2000, 2000, 500, 100])
#stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000, 4500, 3500, 5500, 3500, 4000])

files = ["igor.txt", "ankita.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt", "chris1.txt", "stefan.txt", "ben.txt", "markus.txt", "markusSchnell.txt"]
start = np.array([600, 300, 100, 100, 100, 3500, 500, 2000, 2000, 500, 100])
stop = np.array([3400, 1800, 1700, 1600, 3000, 6000, 4500, 3500, 5500, 3500, 4000])
numberOfClasses = 8

fileLabels = np.array([0, 1, 2, 3, 4, 1, 0, 3, 5, 6, 7, 7])

usedSensors = np.array([0,1,2,3,4,5,6,7,8,9])

print("Using following sensors: ", usedSensors)

########################################################################################################################
trainFrac = 2/3
numDomCoeffs = 20
numDomFreqs = 0
windowWidth = 50
windowShift = 10
numOfSensors = np.size(usedSensors)
########################################################################################################################

dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=False, checkData=False, selectSensors=usedSensors)

for i in range(len(dataSet)):
    dataSet[i] = scaleData(dataSet[i], scaleAmp=True, scaleMean=False, scaleVar=True)

dataWindows = []
numberOfWindows = []
for i in range(len(dataSet)):
    windows, numOfWindows = sliceAndWindowV3(data=dataSet[i], windowWidth=windowWidth, windowShift=windowShift, enaCheck=False, window='tukey', alpha=0.1, enaCWF=0)
    dataWindows.append(windows)
    numberOfWindows.append(numOfWindows)
dataWindows = np.array(dataWindows)
numberOfWindows = np.array(numberOfWindows)
print("Number of windows per dataset: ", numberOfWindows)
#files[i]=="chris.txt"
features = []
labels = []
trainingFeatures = []
testFeatures = []
trainingLabels = []
testLabels = []
for i in range(len(dataWindows)):
    index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
    #random.shuffle(index)
    if ((i==101)|(i==110)):
        print("Dataset:", files[i], " with label:", fileLabels[i], " is for test only...")
        for j in range(numberOfWindows[i]):
            f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, sampleT=0.0165, wavelet = 'haar')
            testFeatures.append(f.T)
            testLabels.append(fileLabels[i])
    else:
        for j in range(numberOfWindows[i]):
            f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, sampleT=0.0165, wavelet = 'haar')
            if j>int(trainFrac*numberOfWindows[i]):
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

with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

with open('my_dumped_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)

## test with normal data:
prediction = []
error = 0
classError = np.zeros(numberOfClasses)
numberOfTestsPerClass = np.zeros(numberOfClasses)
confMat = np.zeros([numberOfClasses, numberOfClasses])
for i in range(len(testLabels)):
    pred = gnb_loaded.predict(testFeatures[i].reshape(1, -1))
    if testLabels[i]==3:
        print(pred)
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

if 0 :
    windowWidth = 100
    windowShift = 10
    numOfSensors = 10
    ser = serial.Serial(port="/dev/ttyUSB0", baudrate=57600)
    dummy = ser.readline() # throw first line
    window = np.zeros([100, numOfSensors])
    count = 0
    while(1):
        count += 1
        line = ser.readline()
        line = line.decode("utf-8")
        line = line.split(",")[:numOfSensors]
        window = np.roll(window, -1, 0)
        window[-1, ::] = line
        if count>=windowShift:
            f = extractSpectralFeatures(dataWindow=window, numDomCoeffs=numDomCoeffs,
                                    numDomFreqs=numDomFreqs, sampleT=0.0165, wavelet='haar')
            count = 0
            pred = clf.predict(f.reshape(1, -1))
            print(pred)
