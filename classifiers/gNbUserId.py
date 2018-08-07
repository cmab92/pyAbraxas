"""
cb, 07.08.2018

 - user identification (with non walking data) using decision tree alg

"""

from abraxas4.abraxasFrame import AbraxasFrame

gNbAbra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=50, windowShift=50, numFreqs=1, numCoeffs=1,
                 enaStatFeats=True, featNormMethod='minmax', trainFraction=2/3, waveletLvl1=False, randomSortTT=False,
                 classSortTT=True, corrPeaks=0, enaRawFeats=False)

gNbAbra.setWindowFunction(functionName='tukey', alpha=0.1)

gNbAbra.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')

gNbAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
               className="not walking")
gNbAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
gNbAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
gNbAbra.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
gNbAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
gNbAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
gNbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
gNbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
gNbAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
gNbAbra.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
gNbAbra.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
gNbAbra.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
gNbAbra.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
gNbAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
gNbAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
gNbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

gNbAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1,
                     className="ankita")
gNbAbra.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

gNbAbra.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2,
                     className="chris")
gNbAbra.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2)
gNbAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
gNbAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
gNbAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
gNbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
gNbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
gNbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
gNbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

gNbAbra.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3,
                     className="crooked")

gNbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4,
                     className="ben")

gNbAbra.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5,
                     className="markus")

gNbAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6,
                     className="igor")
gNbAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

gNbAbra.readDataSet(equalLength=False, checkData=False)

gNbAbra.initFeatNormalization(dumpName="throwAway")
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
gNbAbra.trainClassifier(classifier=clf)
gNbAbra.testClassifier()

