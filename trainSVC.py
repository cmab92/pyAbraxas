import numpy as np
from sklearn import svm

from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures
from abraxasOne.readSeveralFiles import readSeveralFiles
from abraxasOne.helperFunctions import scaleData

def trainSVC():

    files = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt", "chris1.txt"]
    start = np.array([600, 300, 50, 100, 100, 100, 3500, 500])
    stop = np.array([3400, 1800, 1550, 1700, 1600, 3000, 6000, 4500])
    numberOfClasses = 5

    fileLabels = np.array([0, 1, 2, 3, 4, 1, 0, 3])

    usedSensors = np.array([0,1,2,3,4,5,6,7,8,9])

    print("Using following sensors: ", usedSensors)

    ########################################################################################################################
    testFrac = 0.05
    numDomCoeffs = 20
    numDomFreqs = 20
    windowWidth = 100
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
    files[i]=="chris.txt"
    features = []
    labels = []
    trainingFeatures = []
    testFeatures = []
    trainingLabels = []
    testLabels = []
    for i in range(len(dataWindows)):
        index = np.linspace(0,len(dataWindows[i])-1, len(dataWindows[i]))
        #random.shuffle(index)
        if ((i==17)|(i==110)):
            print("Dataset:", files[i], " with label:", fileLabels[i], " is for test only...")
            for j in range(numberOfWindows[i]):
                f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, sampleT=0.0165, wavelet = 'haar')
                testFeatures.append(f.T)
                testLabels.append(fileLabels[i])
        else:
            for j in range(numberOfWindows[i]):
                f = extractSpectralFeatures(dataWindow=dataWindows[i][int(index[j])], numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, sampleT=0.0165, wavelet = 'haar')
                if j>int(testFrac*numberOfWindows[i]):
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
    return clf