from abraxas4.abraxasFrame import AbraxasFrame
import numpy as np

oc = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=2**8, windowShift=2**5, numFreqs=0, numCoeffs=0,
                 enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False, randomSortTT=False,
                 classSortTT=True, corrPeaks=0, enaRawFeats=False, statStages=4)

oc.setWindowFunction(functionName='tukey', alpha=0.9)

oc.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')
# oc.selectSensorSubset(selectedSensors=[], sensorType='fr')
# oc.selectSensorSubset(selectedSensors=[0], sensorType='ir')

oc.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=1500, label=0,
               className="walking")
oc.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=0)

oc.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=0)
oc.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=0)

oc.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=0)
oc.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=0)
oc.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=300, stopTime=1700, label=0)

oc.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=0)

oc.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=0)

oc.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=0)

oc.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=0)

oc.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=1,
               className="not walking")
oc.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=1)
oc.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=1)
oc.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=1)
oc.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=1)
oc.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=1)
oc.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=1)
oc.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=1)
oc.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=1)
oc.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
oc.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
oc.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)
oc.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=1)

oc.readDataSet(equalLength=False, checkData=False)
oc.initFeatNormalization()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

oc.trainClassifier(classifier=model)
oc.testClassifier()

