"""
cb, 07.08.2018

 - dump different dataset for certain classification problems

"""

from abraxas4.abraxasFrame import AbraxasFrame

xgbAbra = False
dtAbra = True

if xgbAbra:
    xgbAbra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numCoeffs=5, numFreqs=1,
                     enaStatFeats=True, wavelet='haar', waveletLvl1=False, featNormMethod='stand', trainFraction=0.66,
                     classSortTT=True, randomSortTT=False, lineThresholdAfterNorm=10, enaRawFeats=False, corrPeaks=2)

    xgbAbra.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')

    xgbAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
                   className="not walking")
    xgbAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
    xgbAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
    xgbAbra.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
    xgbAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
    xgbAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
    xgbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    xgbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
    xgbAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
    xgbAbra.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    xgbAbra.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    xgbAbra.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    xgbAbra.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    xgbAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    xgbAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
    xgbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

    xgbAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1,
                   className="ankita")
    xgbAbra.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

    xgbAbra.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2,
                   className="chris")
    xgbAbra.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2)
    xgbAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
    xgbAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
    xgbAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
    xgbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
    xgbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
    xgbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
    xgbAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

    xgbAbra.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3,
                   className="crooked")

    xgbAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4,
                   className="ben")

    xgbAbra.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5,
                   className="markus")

    xgbAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6,
                   className="igor")
    xgbAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

    xgbAbra.readDataSet(checkData=False, equalLength=False)

    xgbAbra.dumpTeTrData(dumpName='xgbAbra.pkl')

if dtAbra:
    dtAbra = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=25, numFreqs=1, numCoeffs=0,
                     enaStatFeats=True, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False, randomSortTT=False,
                     classSortTT=True, corrPeaks=0, enaRawFeats=False)

    dtAbra.setWindowFunction(functionName='tukey', alpha=0.1)

    dtAbra.selectSensorSubset(selectedSensors=[False, True, True], sensorType='bno')

    dtAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
                   className="not walking")
    dtAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
    dtAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
    dtAbra.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
    dtAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
    dtAbra.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
    dtAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    dtAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
    dtAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
    dtAbra.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    dtAbra.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    dtAbra.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    dtAbra.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    dtAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    dtAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
    dtAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

    dtAbra.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1,
                        className="ankita")
    dtAbra.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

    dtAbra.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2,
                        className="chris")
    dtAbra.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2,
                   className="chris")
    dtAbra.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
    dtAbra.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
    dtAbra.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
    dtAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
    dtAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
    dtAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
    dtAbra.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

    dtAbra.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3,
                        className="crooked")

    dtAbra.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4,
                        className="ben")

    dtAbra.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5,
                        className="markus")

    dtAbra.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6,
                        className="igor")
    dtAbra.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)

    dtAbra.readDataSet(equalLength=False, checkData=False)

    dtAbra.dumpTeTrData(dumpName="dtAbra.pkl")

if False:
    a = AbraxasFrame(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=25, numCoeffs=0, numFreqs=0,
                 kernel='rbf', enaStatFeats=True, wavelet='haar', waveletLvl1=False, featNormMethod='stand',
                 trainFraction=0.66, classSortTT=True, randomSortTT=False, lineThresholdAfterNorm=10, enaRawFeats=False,
                 corrPeaks=2)

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

    b.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3550, stopTime=3800, label=0,
                   className="not walking")
    b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=300, stopTime=500, label=0)
    b.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=0)
    b.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4100, stopTime=4300, label=0)
    b.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=0)
    b.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=0)
    b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7100, stopTime=8000, label=0)
    b.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5200, stopTime=6000, label=0)
    b.addDataFiles(fileSourceName="novcc.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    b.addDataFiles(fileSourceName="nowalk.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    b.addDataFiles(fileSourceName="nowalk2.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    b.addDataFiles(fileSourceName="nowalk3.txt", fileSourcePath="../", startTime=0, stopTime=10000, label=0)
    b.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=0)
    b.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=9000, stopTime=12000, label=0)
    b.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=24000, stopTime=26000, label=0)

    b.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=1)
    b.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=1)

    # b.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=2)
    b.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=2,
                   className="chris")
    b.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=2)
    b.addDataFiles(fileSourceName="chris2.txt", fileSourcePath="../", startTime=1000, stopTime=8600, label=2)
    b.addDataFiles(fileSourceName="chrisOut.txt", fileSourcePath="../", startTime=1000, stopTime=9000, label=2)
    b.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=1000, stopTime=4000, label=2)
    b.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=4250, stopTime=5250, label=2)
    b.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=6000, stopTime=14000, label=2)
    b.addDataFiles(fileSourceName="chrisOut2.txt", fileSourcePath="../", startTime=14000, stopTime=22000, label=2)

    b.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1600, label=3)

    b.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=4)

    b.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=3300, label=5)

    b.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=6)
    b.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=6)