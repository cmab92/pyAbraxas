import os 
import sys
sys.path.append('../')
from abraxasThree.classifierClass import AbraxasClassifier

# user identification:

a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=25, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='stand', kernel='rbf', trainFraction=2/3, waveletLvl1=True,
                      randomSortTT=False, classSortTT=True)

a.setWindowFunction(functionName='tukey', alpha=0.9)
# a.plotWindowFunction()

a.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
a.selectSensorSubset(selectedSensors=[0, 1, 2, 4, 6, 7, 8], sensorType='ir')

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

a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1)
a.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

# a.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2)
a.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2,
               className="chris")
a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)

a.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3)

a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4)

a.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5)

a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6)
a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

a.readDataSet(equalLength=False, checkData=False)


useDump = True

if useDump:
    a.loadDumpNormParam(dumpName="dataOnly")
    clf = a.loadDumpClassifier("dataOnly")
    a.testClassifier(classifier=clf)
    a.setFileSink(fileSinkName="chris", fileSinkPath="../")
    a.startLiveClassification()
else:
    a.initFeatNormalization(dumpName="dataOnly")
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    a.trainClassifier(classifier=clf)
    a.dumpClassifier(dumpName="dataOnly")
    a.testClassifier()

