import numpy as np
from abraxasThree.classifierClass import AbraxasClassifier
from sklearn import svm
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=50, numFreqs=3, numCoeffs=10,
                      enaStatFeats=False, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True, enaRawFeats=False, corrPeaks=2)

a.setWindowFunction(functionName='tukey', alpha=0.9)
# a.plotWindowFunction()

a.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')
# a.selectSensorSubset(selectedSensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], sensorType='ir')
# a.selectSensorSubset(selectedSensors=[0, 1], sensorType='fr')

a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
               className="not walking")
a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
a.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
a.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
a.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1,
               className="ankita")
a.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

# a.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2)
a.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2,
               className="chris")
a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
a.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
a.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

a.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3,
               className="crooked")
a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4,
               className="ben")

a.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5,
               className="markus")

a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6,
               className="igor")
a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

a.readDataSet(equalLength=False, checkData=False)
windows, labels = a.windowSplitSourceDataTT()
features = a.initFeatNormalization(windows)

# create the RFE model and select 3 attributes

features = np.array(features)
labels = np.array(labels)

model = XGBClassifier()
model = RFE(model, 100)
model.fit(features, labels)
# summarize the selection of the attributes
print(model.support_)
print(model.ranking_)
fov = a.returnFeatureIndices()
print(fov)
plt.scatter(fov, model.ranking_)
plt.figure()
for i in range(17):
    print(i, ":", np.sum(model.support_[fov==i]))
plt.scatter(fov, model.support_)
plt.show()
# print(model.feature_importances_)
# plt.plot(model.feature_importances_)
# plt.show()

