from abraxas4.abraxasFrame import AbraxasFrame
import numpy as np
import random
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#abra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numCoeffs=5, numFreqs=1,
#                     enaStatFeats=True, wavelet='haar', waveletLvl1=False, featNormMethod='stand', trainFraction=0.66,
#                     classSortTT=True, randomSortTT=False, lineThresholdAfterNorm=10, enaRawFeats=False, corrPeaks=2)

abra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numFreqs=1, numCoeffs=5,
                      enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True, corrPeaks=1, enaRawFeats=False)

abra.setWindowFunction(functionName='tukey', alpha=0.3)
abra.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')

abra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
                     className="not walking")
abra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
abra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
abra.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
abra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
abra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
abra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
abra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
abra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
abra.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
abra.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
abra.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
abra.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
abra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
abra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
abra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

abra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1,
                     className="ankita")
abra.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

abra.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2,
                     className="chris")
abra.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2)
abra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
abra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
abra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
abra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
abra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
abra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
abra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

abra.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3,
                     className="crooked")

abra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4,
                     className="ben")

abra.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5,
                     className="markus")

abra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6,
                     className="igor")
abra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

abra.readDataSet(checkData=False, equalLength=True)

windowedData, windowLabels = abra.windowSplitSourceDataTT()

index = np.linspace(0, len(windowedData) - 1, len(windowedData), dtype=int)
random.shuffle(index)

trainingData = []
trainingLabels = []
testData = []
testLabels = []
for i in range(int(len(windowedData))):
    if i/len(windowedData) < 0.8:
        trainingData.append(windowedData[index[i]])
        trainingLabels.append(windowLabels[index[i]])
    else:
        testData.append(windowedData[index[i]])
        testLabels.append(windowLabels[index[i]])

trainingData = abra.initFeatNormalization(inputData=trainingData)

for i in range(len(testData)):
    testData[i] = abra.featureNormalization(features=abra.extractFeatures(data=testData[i]), initDone=True)

#model = xgboost.XGBClassifier(max_depth=3, learning_rate=0.3)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
kfold = StratifiedKFold(n_splits=3)
results = cross_val_score(model, np.array(trainingData), np.array(trainingLabels), cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
