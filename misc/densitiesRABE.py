import numpy as np
import matplotlib.pyplot as plt
from abraxasThree.classifierClass import AbraxasClassifier
from misc.kernelDensityEstimator import kernelDensityEstimator

a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=100, numFreqs=0, numCoeffs=0,
                      enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)

a.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
# a.selectSensorSubset(selectedSensors=[0, 2, 4, 6, 8], sensorType='ir')
# a.selectSensorSubset(selectedSensors=[], sensorType='fr')

#a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
#                className="not walking")
# a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
# a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
# a.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
# a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
# a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
# a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
# a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
# a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
a.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
a.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
# a.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
# a.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
# a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1)
a.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

a.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=1)
a.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=1,
               className="chris")
# a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=1)
# a.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=1)
# a.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=1)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=1)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=1)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=1)
a.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=1)

# a.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=1)

# a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=1)

# a.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=1)

# a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=1)
# a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=1)

dataSet = a.readDataSet(equalLength=False, checkData=False)
windowData, windowLabels = a.windowSplitSourceDataTT(inputData=dataSet)

features = a.initFeatNormalization(inputData=windowData)
features = a.featureNormalization(features=features, initDone=True)

features = np.array(features)
windowLabels = np.array(windowLabels)

features0 = features[windowLabels==0]
features1 = features[windowLabels==1]

m0 = []
m1 = []
diff = []
for i in range(int(len(features[0, ::]))):
    #f = kernelDensityEstimator(x=features0[::, f0], h=0.15)
#    plt.scatter(i*np.ones(np.size(features0[::, f0])), features0[::, f0], c='r', marker="s", alpha=0.5)
    # m0.append(kernelDensityEstimator(features0[::, f0], h=0.15))
    #plt.plot(f, 'r')
    #f = kernelDensityEstimator(x=features1[::, f0], h=0.15)
    #plt.plot(f, 'b')
#    plt.scatter(i*np.ones(np.size(features1[::, f0])), features1[::, f0], c='b', marker="*", alpha=0.5)
    # m1.append(kernelDensityEstimator(features1[::, f0], h=0.15))
    f0 = kernelDensityEstimator(features0[::, i], h=0.15)
    f1 = kernelDensityEstimator(features1[::, i], h=0.15)
    diff.append(np.array(f0))
    diff.append(np.array(f0))
    diff.append(np.array(f0))
    diff.append(np.array(f0))
    diff.append(np.array(f1))
    diff.append(np.array(f1))
    diff.append(np.array(f1))
    diff.append(np.array(f1))
plt.imshow(diff)
plt.colorbar()
plt.show()
#plt.plot(dataSet[0][::, 0])
#plt.plot(dataSet[0][::, 1])
#plt.show()

