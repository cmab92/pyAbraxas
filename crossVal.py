import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures
from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.helperFunctions import scaleData
from abraxasOne.helperFunctions import shuffleData
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues
########################################################################################################################
files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt"]
dataLabels = np.array([0,1,2,3,4,1,0])
start = np.array([600, 300, 50, 100, 100, 100, 3500])
stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000])
########################################################################################################################
usedSensors = np.array([0,1,2,3,4,5,6,7,8,9])
print("Using following sensors: ", usedSensors)
########################################################################################################################
## load dataset:
########################################################################################################################
dataSet = readSeveralFiles(files=files, startTimes=start, stopTimes=stop, path="", numberOfIrSensors=10, numberOfForceSensors=2, equalLength=False, checkData=False, selectSensors=usedSensors)
########################################################################################################################
## normalize data
########################################################################################################################
for i in range(len(dataSet)):
    dataSet[i] = scaleData(dataSet[i], scaleAmp=True, scaleMean=False, scaleVar=True)
########################################################################################################################
## extract windows:
########################################################################################################################
dataWindows = []
numberOfWindows = []
for i in range(len(dataSet)):
    windows, numOfWindows = sliceAndWindowV3(data=dataSet[i], windowWidth=20, windowShift=10, enaCheck=False, window='tukey', alpha=0.2, enaCWF=1)
    dataWindows.append(windows)
    numberOfWindows.append(numOfWindows)
dataWindows = np.array(dataWindows)
numberOfWindows = np.array(numberOfWindows)
print("Number of windows per dataset: ", numberOfWindows)
########################################################################################################################
## feature extraction:
########################################################################################################################
features = []
labels = []
trainingFeatures = []
testFeatures = []
trainingLabels = []
testLabels = []
for i in range(len(dataWindows)):
    index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
    random.shuffle(index)
    for j in range(numberOfWindows[i]):
        f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=6, numDomFreqs=3, sampleT=0.0165, wavelet = 'haar')
        if j>int(2/3*numberOfWindows[i]):
            testFeatures.append(f.T)
            testLabels.append(dataLabels[i])
        else:
            trainingFeatures.append(f.T)
            trainingLabels.append(dataLabels[i])
features = np.concatenate([trainingFeatures, testFeatures])
labels = np.concatenate([trainingLabels, testLabels])
########################################################################################################################
## simply train and test svm:
########################################################################################################################
if 0:
    clf = svm.SVC(kernel='rbf')#, C=2.236, gamma=0.00227)
    clf.fit(trainingFeatures, trainingLabels)
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
import pandas as pd
#
#data = {'0': confMat[::,0], '1': confMat[::,1], '2': confMat[::,2], '3': confMat[::,3], '4': confMat[::,4]}
#output_df = pd.DataFrame(data=data)
#output_df.to_csv("/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/" + "confMat.dat", sep=' ', index=False, header=False)
########################################################################################################################
## Xval svm:
########################################################################################################################
clf = svm.SVC(kernel='rbf')
scores = cross_val_score(clf, np.concatenate([testFeatures, trainingFeatures]),  np.concatenate([testLabels, trainingLabels]), cv=5)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
########################################################################################################################
## grid search:
########################################################################################################################
#clf = svm.SVC(kernel='rbf', C=2.236, gamma=0.00227)
#Cs = np.linspace(2.23, 2.25, 100)
#print(Cs)
#clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs))
#clf.fit(trainingFeatures, trainingLabels)
#print("best score: ", clf.best_score_)
#print("best estimator: ", clf.best_estimator_)
## Prediction performance on test set is not as good as on train set
#scores = cross_val_score(clf, features,  labels, cv=5)
#print("X-val. accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))