"""
cb, 07.07.2018

 - train support vector classifier and dump
"""
import numpy as np
import pickle
from sklearn import svm
from abraxasTwo.extractFeatures import extractFeatures
import matplotlib.pyplot as plt


def trainSVC(featureMatrix, labels, kernel=None, C=None, gamma=None):

    if kernel is not None and gamma is not None and C is not None:
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    elif kernel is not None and gamma is not None and C is None:
        clf = svm.SVC(kernel=kernel, gamma=gamma)
    elif kernel is not None and gamma is None and C is None:
        clf = svm.SVC(kernel=kernel)
    elif kernel is not None and gamma is None and C is not None:
        clf = svm.SVC(kernel=kernel, C=C)
    elif kernel is None and gamma is not None and C is not None:
        clf = svm.SVC(gamma=gamma, C=C)
    elif kernel is None and gamma is not None and C is None:
        clf = svm.SVC(gamma=gamma)
    elif kernel is None and gamma is None and C is not None:
        clf = svm.SVC(C=C)
    else:
        clf = svm.SVC()

    clf.fit(featureMatrix, labels)

    svcDump = pickle.dumps(clf)
    classifier = pickle.loads(svcDump)
    return classifier


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet
    from abraxasTwo.windowDataSet import windowDataSet
    from abraxasTwo.getFeatures import getFeatures

    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2,
                          selBnoData=[False, False, False], checkData=False, equalLength=True, rawData=None)
    print("Data read...")
    windows, labels = windowDataSet(dataSet, labels=[0, 1, 2, 3, 4, 1, 0], width=100, shift=10)
    print("... windowed ....")
    features, normVals = getFeatures(windows, numCoeffs=10, numFreqs=10, normalize='stand')
    print("... and features extracted.")

    features = np.array(features)
    labels = np.array(labels)

    import random
    index = np.linspace(0, len(features)-1, len(features), dtype=int)
    random.shuffle(index)
    trainIndex = index[0:int(2/3*len(index))]
    testIndex = index[int(2/3*len(index)):]

    trainingData = features[trainIndex, ::]
    trainingLabels = labels[trainIndex]

    testData = features[testIndex, ::]
    testLabels = labels[testIndex]
    plt.hist(labels[testIndex])
    plt.title("test")
    plt.figure()
    plt.hist(labels[trainIndex])

    print("Training...")
    clf = trainSVC(featureMatrix=trainingData, labels=trainingLabels)
    print("...done!")

    errors = np.array([0, 0, 0, 0, 0])
    occurrenceCount = np.array([0, 0, 0, 0, 0])

    for i in range(len(testIndex)):
        occurrenceCount[int(labels[testIndex[i]])] += 1
        # f = (features[testIndex[i], ::] - normVals[0, ::])/(normVals[1, ::] - normVals[0, ::])
        f = (features[testIndex[i], ::] - normVals[0, ::]) / normVals[1, ::]
        if labels[testIndex[i]] != clf.predict(f.reshape(1, -1)):
            errors[int(labels[testIndex[i]])] += 1

    print(errors)
    print(occurrenceCount)
    print(errors/occurrenceCount)
    plt.show()

