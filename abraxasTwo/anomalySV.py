"""
cb, 10.07.2018

 - simple one class svm

"""

import multiprocessing
from six.moves import cPickle
from abraxasTwo.receiveData import receiveData
from abraxasTwo.extractFeatures import extractFeatures
from abraxasTwo.normalize import normalize


def anomalySV(classifier, features, param):
    f = features.get()
    prediction = classifier.predict(f.reshape(1, -1))
    print(prediction)


if __name__ == '__main__':

    with open('dumpedSVC.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)
    with open('dumpedParams.pkl', 'rb') as parameter:
        normParams = cPickle.load(parameter)

    dataWindow = multiprocessing.Queue()
    features = multiprocessing.Queue()

    dataReception = multiprocessing.Process(target=receiveData, args=(10, 2, "../", "test.txt", None, 57600, 100, 10,
                                                                      "tukey", 0.1, dataWindow, 0, ))

    featureExtraction = multiprocessing.Process(target=extractFeatures, args=(dataWindow, 10,
                                                                       10, False, True, 'haar', True, features, ))

    classification = multiprocessing.Process(target=anomalySV, args=(classifier, features))

    dataReception.start()
    featureExtraction.start()
    classification.start()

