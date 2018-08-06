from abraxasThree.classifierClass import AbraxasClassifier

a = AbraxasClassifier(numIrSensors=1, numFrSensors=0, windowWidth=1500, windowShift=10, numFreqs=0, numCoeffs=0,
                      enaStatFeats=False, featNormMethod='stand', trainFraction=2/3, waveletLvl1=False,
                      randomSortTT=False, classSortTT=True)

a.setFileSink(fileSinkName="test.txt", fileSinkPath="../")
a.setWindowFunction(functionName='rect', alpha=0)
a.setupSerialInterface(port="/dev/ttyACM0", baudRate=57600)
a.startReceiveData()
a.startPlotStreamData(sensorNr=[5, 7, 8, 10])