"""
cb, 07.07.2018

 - train support vector classifier and dump
"""
import numpy as np
import pickle
from sklearn import svm
from abraxasTwo.extractFeatures import extractFeatures
import matplotlib.pyplot as plt


def trainSVC(dataWindows, windowLabels, numCoeffs, numFreqs, kernel=None, C=None, gamma=None):

    featureVectors = []

    for i in range(len(dataWindows)):
        featureVectors.append(extractFeatures(dataWindows[i], numCoeffs=numCoeffs, numFreqs=numFreqs))
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
        clf = svm.SVC(gama=gamma)
    elif kernel is None and gamma is None and C is not None:
        clf = svm.SVC(C=C)
    elif kernel is None and gamma is None and C is None:
        clf = svm.SVC()



    clf.fit(featureVectors, windowLabels)

    svcDump = pickle.dumps(clf)
    classifier = pickle.loads(svcDump)
    return classifier


if __name__ == '__main__':
    from abraxasTwo.readDataSet import readDataSet

    fileNames = ["igor.txt", "ankita.txt", "chris_asymm.txt", "chris_pos2.txt", "chris_c.txt", "ankita_pos2_lrRl.txt",
                 "igor2.txt"]
    filePath = "../"
    dataSet = readDataSet(fileNames=fileNames, filePath=filePath, nrIrSensors=10, nrFrSensors=2,
                          selBnoData=[False, False, False], checkData=False, equalLength=False, rawData=None)
    from abraxasTwo.windowDataSet import windowDataSet
    windows, labels = windowDataSet(dataSet, labels=[0, 1, 2, 3, 4, 1, 0], width=500, shift=500)
    trainSVC(dataWindows=windows, windowLabels=labels, numCoeffs=10, numFreqs=10)

