import os 
import sys
sys.path.append('../')
from abraxasThree.classifierClass import AbraxasClassifier

# user identification:

a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=25, numFreqs=1, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='stand', kernel='rbf', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True, corrPeaks=0, enaRawFeats=True)

a.setWindowFunction(functionName='tukey', alpha=0.2)
#a.plotWindowFunction()

a.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
# a.selectSensorSubset(selectedSensors=[0, 1, 2, 3], sensorType='ir')
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

a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1)
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

a.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3)

a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4)

a.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5)

a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6)
a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

a.readDataSet(equalLength=False, checkData=False)


useDump = False

if useDump:
    a.loadDumpNormParam(dumpName="KNeighborsClassifier")
    clf = a.loadDumpClassifier("KNeighborsClassifier")
    a.testClassifier(classifier=clf)
    a.setFileSink(fileSinkName="chris", fileSinkPath="../")
    a.startLiveClassification()
else:
    a.initFeatNormalization(dumpName="KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    a.trainClassifier(classifier=clf)
    a.dumpClassifier(dumpName="KNeighborsClassifier")
    a.testClassifier()

