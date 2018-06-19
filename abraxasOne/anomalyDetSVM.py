########################################################################################################################
## cb, 05.04.18
##
##
## todo: optimize normalization factor (loadAndPrepareData)
## todo: reasonable choice and optimization of window size
## todo: reasonable choice and optimization of window function
## todo: introduce and optimize normalization factors for wavelet and fourier features (consider sampleT)
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures
from abraxasOne.sliceAndWindow import sliceAndWindowV3 as sliceAndWindow
from abraxasOne.loadAndClean import loadAndClean
from abraxasOne.readSerialFile import *
from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.helperFunctions import scaleData
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
##

files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"]
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])

usedSensors = np.array([0,1,2,3,4,5,6,7,8,9])
print("Using following sensors: ", usedSensors)

dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="../", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=False, checkData=False, selectSensors=usedSensors)

for i in range(len(dataSet)):
    dataSet[i] = scaleData(dataSet[i], scaleAmp=True, scaleMean=False, scaleVar=True)

dataWindows = []
numberOfWindows = []
for i in range(len(dataSet)):
    windows, numOfWindows = sliceAndWindow(data=dataSet[i], windowWidth=100, windowShift=10, enaCheck=False, window='tukey', alpha=0.1, enaCWF=0)
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
outlierFeatures = []

numOfWindowsTeD = 0
numOfWindowsOuD = 0
numOfWindowsTrD = 0

for i in range(len(dataWindows)):
    index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
    random.shuffle(index)
    for j in range(numberOfWindows[i]):
        f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=3, numDomFreqs=4, sampleT=0.0165, wavelet = 'haar')
        if j>int(0.95*numberOfWindows[i]):
            outlierFeatures.append(np.random.rand(np.size(f)))
            numOfWindowsOuD += 1
        elif (j<int(0.95*numberOfWindows[i])) & (j>int(0.8*numberOfWindows[i])):
            testFeatures.append(f)
            numOfWindowsTeD += 1
        else:
            trainingFeatures.append(f)
            numOfWindowsTrD += 1
if 1:
    ## train svm:
    clf = svm.OneClassSVM(kernel="poly", nu=0.01, gamma=0.25)
    print("Training:")
    print("size:", np.shape(trainingFeatures))
    clf.fit(trainingFeatures)

    ## test with normal data:
    print("Test:")
    print("size:", np.shape(testFeatures))
    predTest = clf.predict(testFeatures)
    ## test with non-walking data:
    print("Outlier:")
    print("size:", np.shape(testFeatures))
    predOutlier = clf.predict(outlierFeatures)
    ## show errors:
    n_error_test = predTest[predTest == -1].size
    n_error_outliers = predOutlier[predOutlier == 1].size
    print('Error on test data:')
    print(str(n_error_test/numOfWindowsTeD*100)+'%')
    print('Error on outlier data:')
    print(str(n_error_outliers/numOfWindowsOuD*100)+'%')
