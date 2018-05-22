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
from sklearn import svm
from extractSpectralFeatures import extractSpectralFeatures
from sliceAndWindow import sliceAndWindowV2 as sliceAndWindow
from loadAndClean import loadAndClean
from readSerialFile import *
##
TSAMPLE = 0.0165 # in seconds (determines window width)
irData, forceData, quatData, linAccData, angVecData = loadAndClean(fileName="_20185161226.txt", numberOfIrSensors=10, numberOfForceSensors=4, tSample=TSAMPLE)
data = []
for i in range(10):
    data.append(irData[i])
for i in range(2):
    data.append(forceData[i])
## get windows:
WSIZE = 0.165 # window size in seconds
ENASC = 1 # slice check
WINDOWFCT = 'ham'
ALPHA = 10
ENAWC = 0 ## window function check
WINDOWSHIFT = 0.01
trainingData, numOfWindowsTrD = sliceAndWindow(data=data, startT=0, stopT=10**9, windowWidth=WSIZE, windowShift=WINDOWSHIFT, sampleT=TSAMPLE, enaCheck=ENASC, window=WINDOWFCT, alpha=ALPHA, enaCWF=ENAWC)
trainingData, numOfWindowsTrD = sliceAndWindow(data=data, startT=40, stopT=80, windowWidth=WSIZE, windowShift=WINDOWSHIFT, sampleT=TSAMPLE, enaCheck=ENASC, window=WINDOWFCT, alpha=ALPHA)
print('Number of windows (Training Data)')
print(numOfWindowsTrD)
testData, numOfWindowsTeD = sliceAndWindow(data=data, startT=122, stopT=130, windowWidth=WSIZE, windowShift=WINDOWSHIFT, sampleT=TSAMPLE, enaCheck=ENASC, window=WINDOWFCT, alpha=ALPHA)
print('Number of windows (Test Data)')
print(numOfWindowsTeD)
outlierData, numOfWindowsOuD = sliceAndWindow(data=data, startT=118, stopT=121, windowWidth=WSIZE, windowShift=WINDOWSHIFT, sampleT=TSAMPLE, enaCheck=ENASC, window=WINDOWFCT, alpha=ALPHA)
print('Number of windows (Outlier Data)')
print(numOfWindowsOuD)
## get features:
NDOMCOEFFS = 1
NDOMFREQS = 2
WAVELET = 'db32'
trainingFeatures = []
testFeatures = []
outlierFeatures = []
for i in range(numOfWindowsTrD):
    trainingFeatures.append(extractSpectralFeatures(dataWindow=trainingData[i], numDomCoeffs=NDOMCOEFFS, numDomFreqs=NDOMFREQS, sampleT=TSAMPLE, wavelet=WAVELET))
plt.show()
for i in range(numOfWindowsTeD):
    testFeatures.append(extractSpectralFeatures(dataWindow=testData[i], numDomCoeffs=NDOMCOEFFS, numDomFreqs=NDOMFREQS, sampleT=TSAMPLE, wavelet=WAVELET))
plt.show()
for i in range(numOfWindowsOuD):
    outlierFeatures.append(extractSpectralFeatures(dataWindow=outlierData[i], numDomCoeffs=NDOMCOEFFS, numDomFreqs=NDOMFREQS, sampleT=TSAMPLE, wavelet=WAVELET))
plt.show()
## train svm:
clf = svm.OneClassSVM(kernel="rbf", nu=0.01, gamma=0.25)
clf.fit(trainingFeatures)
predTrain = clf.predict(trainingFeatures)
## test with normal data:
predTest = clf.predict(testFeatures)
## test with non-walking data:
predOutlier = clf.predict(outlierFeatures)
## show errors:
n_error_train = predTrain[predTrain == -1].size
n_error_test = predTest[predTest == -1].size
n_error_outliers = predOutlier[predOutlier == 1].size
print('Error on test data:')
print(str(n_error_test/numOfWindowsTeD*100)+'%')
print('Error on outlier data:')
print(str(n_error_outliers/numOfWindowsOuD*100)+'%')
#for i in range(1):
 #   if (predOutlier[i]==1):
  #      plt.figure()
   #     plt.plot(outlierData[i])
    #    plt.title('outlier'+str(i))
