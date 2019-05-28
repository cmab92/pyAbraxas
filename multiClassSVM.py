import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from six.moves import cPickle
from abraxasOne.splitDataTrainTest import splitDataTrainTest
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues

files = ["igor.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt", "chris1.txt", "chris_pos2.txt", "ben.txt", "markus.txt", "markusSchnell.txt"]
start = np.array([600, 100, 100, 3500, 500, 100, 2000, 500, 100])
stop = np.array([3400, 1600, 3000, 6000, 4500, 1400, 5500, 3500, 1400])
numberOfClasses = 6

fileLabels = np.array([0, 1, 2, 0, 3, 3, 4, 5, 5])
fileLabelsSym = ['igor, 0', 'crooked, 1', 'ankita, 2', 'chris, 3', 'ben, 4', 'markus, 5']

usedSensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

trF, teF, trL, teL, mean, std_dev = splitDataTrainTest(files, start, stop, fileLabels, windowWidth=100, windowShift=10,
                                                       numDomCoeffs=10, numDomFreqs=0, trainFrac=1, statFeat=True,
                                                       shuffleData=False, checkData=False)

normalizationData = []
normalizationData.append(mean)
normalizationData.append(std_dev)
normalizationData = np.array(normalizationData)
np.savetxt("normData.txt", normalizationData, delimiter=",")

clf = svm.SVC(kernel='rbf')#, C=100.0236, gamma=10.00227)
clf.fit(trF, trL)

with open('dumpedSVC.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

prediction = []
error = 0
classError = np.zeros(numberOfClasses)
numberOfTestsPerClass = np.zeros(numberOfClasses)
confMat = np.zeros([numberOfClasses, numberOfClasses])
for i in range(len(teL)):
    pred = clf.predict(teF[i].reshape(1, -1))
    confMat[int(pred),teL[i]] += 1
    numberOfTestsPerClass[teL[i]] += 1
    if pred != teL[i]:
        error += 1
        classError[teL[i]] += 1
confMat = confMat*1/numberOfTestsPerClass
error = error/len(teL)*100
print("Class. Error")
print(error)
print("per class:")
print(classError/numberOfTestsPerClass*100)
plotMatrixWithValues(confMat, title_=str(fileLabelsSym))
