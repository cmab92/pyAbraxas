from abraxas4.abraxasFrame import AbraxasFrame
import numpy as np

oc = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numFreqs=1, numCoeffs=3,
                 enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False, randomSortTT=False,
                 classSortTT=True, corrPeaks=0, enaRawFeats=False, statStages=3)

oc.setWindowFunction(functionName='tukey', alpha=0.1)

oc.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')
# oc.selectSensorSubset(selectedSensors=[], sensorType='fr')
# oc.selectSensorSubset(selectedSensors=[0], sensorType='ir')

oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1500, stopTime=5000, label=0)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=8000, stopTime=13000, label=0)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=15000, stopTime=20000, label=0)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=7500, stopTime=7600, label=1)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=7600, stopTime=7700, label=1)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=5360, stopTime=5460, label=1)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=13600, stopTime=13700, label=1)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14350, stopTime=14550, label=1)
oc.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcekPath="../", startTime=20300, stopTime=20400, label=1)

oc.readDataSet(equalLength=False, checkData=False)
oc.dumpTeTrData(dumpName="anomaly.pkl")

TrainFeat, TrainLabel, TestFeat, TestLabel = oc.loadTeTrDump(dumpName="anomaly.pkl")

data = np.concatenate([TestFeat, TrainFeat])
label = np.concatenate([TestLabel, TrainLabel])

normal = data[label==0]
anomal = data[label==1]

training = normal[0:int(2/3*len(normal))]
test = normal[int(2/3*len(normal))::]

from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel='linear')

model.fit(training)
preds = model.predict(test)
preds = np.reshape(preds, len(preds))
print("False Negatives: ", np.sum(preds==-1)/len(preds))
print("True Positives: ", np.sum(preds==1)/len(preds))
preds = model.predict(anomal)
preds = np.reshape(preds, len(preds))
print("False Positives: ", np.sum(preds==1)/len(preds))
print("True Negatives: ", np.sum(preds==-1)/len(preds))

