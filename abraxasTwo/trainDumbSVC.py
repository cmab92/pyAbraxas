"""
cb, 07.07.2018

 - train support vector classifier and dump

"""
import numpy as np
from six.moves import cPickle
from sklearn import svm
from abraxasTwo.extractFeatures import extractFeatures
import matplotlib.pyplot as plt


def trainDumbSVC(featureMatrix, labels):

    featureMatrix, params = normalize(featureMatrix)

    with open('dumpedParams.pkl', 'wb') as parameter:
        cPickle.dump(params, parameter)

    classifier = svm.SVC()

    classifier.fit(featureMatrix, labels)

    with open('dumpedSVC.pkl', 'wb') as fid:
        cPickle.dump(classifier, fid)

    return classifier


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet
    from abraxasTwo.windowDataSet import windowDataSet
    from abraxasTwo.getFeatures import getFeatures
    from abraxasTwo.normalize import normalize
    from abraxasTwo.miscFunc import plotMatrixWithValues

    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt", "markus.txt", "markusSchnell.txt", "stefan.txt", "ben.txt", "chris1.txt"]
    filePath = "../"

    startT = np.array([100, 200, 200, 100, 100, 150, 600, 500, 100, 500, 2000, 500])
    stopT = np.array([2900, 1900, 1400, 1700, 1700, 2500, 6000, 4000, 4000, 7000, 6000, 5000])

    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2, startTimes=startT,
                          stopTimes=stopT, selBnoData=[False, True, True], checkData=False, equalLength=False,
                          rawData=None, selIrSensors=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                          selFrSensors=np.array([0, 1]))

    print("Data read...")

    classLabels = [0, 1, 2, 3, 4, 1, 0, 5, 6, 7, 8, 4]
    numberOfClasses = len(set(classLabels))

    windows, labels = windowDataSet(dataSet, labels=classLabels, width=100, shift=25, enaCheck=False)

    '''
    data = windows[0]
    label = labels[0]*np.ones(len(windows[0]))
    for i in range(len(windows)-1):
        data = np.concatenate([data, windows[i+1]])
        label = np.concatenate([label, labels[i+1]*np.ones(len(windows[i+1]))])

    plt.plot(data)
    plt.plot(label)
    plt.show()
    '''

    print("... and windowed.")
    features = getFeatures(windows, numCoeffs=10, numFreqs=10, statFeat=True)
    print("... features extracted.")

    '''
    classColor = ['r', 'g', 'b', 'y', 'k']
    classMark = ['x', 'o']
    for i in range(len(features)):
        if labels[i] == 0 or labels[i] == 1:
            plt.scatter(np.linspace(0, len(features[i]-1), len(features[i])), features[i], c=classColor[int(labels[i])],
                        marker=classMark[labels[i]])
    plt.show()
    '''

    features = np.array(features)
    labels = np.array(labels)

    testData = []
    testLabels = []
    trainingData = []
    trainingLabels = []

    for i in range(numberOfClasses):
        count = 0
        for j in range(len(labels)):
            if labels[j] == i:
                if count/np.sum(np.ones(np.size(labels))[labels == i]) > 2/3:
                    testData.append(features[j])
                    testLabels.append(i)
                else:
                    trainingData.append(features[j])
                    trainingLabels.append(i)
                count += 1

    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    testData = np.array(testData)
    testLabels = np.array(testLabels)

    print("Training...")
    clf = trainDumbSVC(featureMatrix=trainingData, labels=trainingLabels)
    print("...done!")

    with open('dumpedParams.pkl', 'rb') as parameter:
        normParams = cPickle.load(parameter)
    print(normParams)

    errors = np.zeros(numberOfClasses)
    occurrenceCount = np.zeros(numberOfClasses)
    confMat = np.zeros([numberOfClasses, numberOfClasses])

    for i in range(len(testLabels)):
        occurrenceCount[int(testLabels[i])] += 1
        testData[i] = normalize(testData[i], param=normParams)
        prediction = clf.predict(testData[i].reshape(1, -1))
        confMat[int(prediction), int(testLabels[i])] += 1

    for i in range(numberOfClasses):
        confMat[::, i] = confMat[::, i]/occurrenceCount[i]

    for i in range(numberOfClasses):
        print("Total number of samples for " + str(i) + "-th class:", np.sum(np.ones(np.size(labels))[labels == i]))
        print("Test sample number for " + str(i) + "-th class:", np.sum(np.ones(np.size(testLabels))[testLabels == i]))

    print("Percentage of correct classification in total:")
    print(np.sum(np.diag(confMat))/numberOfClasses)

    plotMatrixWithValues(confMat)

