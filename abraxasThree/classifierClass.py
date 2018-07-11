"""
cb, 11.07.2018

 - same same but classy

"""
import serial
import datetime
import csv
import pywt
import scipy.signal
import scipy.interpolate
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust import mad


class AbraxasClassifier:

    def __init__(self, numIrSensors, numFrSensors, windowWidth, windowShift, numCoeffs, numFreqs, enaStatFeats,
                 wavelet='haar', wvltLvl1=False, featNormMethod='stand'):

        # Hardware information and parameters
        self.__numIrSensors = numIrSensors
        self.__numFrSensors = numFrSensors
        self.__nrAnalogSensors = self.__numIrSensors + self.__numFrSensors
        self.__frameLength = self.__nrAnalogSensors + 5     # 5 for bno data including index (received format)
        self.__windowLength = self.__nrAnalogSensors + 10   # 10 is for reformatted bno data
        self.__port = None                                # see function receive data
        self.__baudRate = 57600
        self.__sampleT = 0.0165

        # File sources and sinks
        self.__fileSinkName = ""
        self.__fileSinkPath = "../"
        self.fileSourceName = []
        self.fileSourcePath = []
        self.fileSourceStartT = []
        self.fileSourceStopT = []
        self.fileLabels = []

        # Sensor selection
        self.__selIrSensors = np.linspace(0, self.__numIrSensors - 1, self.__numIrSensors)    # select all ir sensors
        self.__selFrSensors = np.linspace(0, self.__numFrSensors - 1, self.__numFrSensors)    # select all fr sensors
        self.__selBnoData = [True, True, True]   # select all bno data {quat, linacc, angvec}

        # Constants and parameters
        self.__windowWidth = windowWidth
        self.__windowShift = windowShift
        self.__numCoeffs = numCoeffs
        self.__numFreqs = numFreqs
        self.__enaStatFeats = enaStatFeats
        self.__wavelet = wavelet
        self.__wvltLvl1 = wvltLvl1
        self.__featNormMethod = featNormMethod
        self.__windowAlpha = 0.1
        self.__windowFunction = scipy.signal.tukey(self.__windowWidth, self.__windowAlpha)
        self.__quatNormFact = 2 ** 14
        self.__linAccNormFact = 2 ** 4
        self.__angVecNormFact = 2 ** 4
        self.__anaNormFact = 1023
        self.__normVal = None

        # Queues and processes
        self.__windowDataQueue = None     # queue for window data
        self.__featureQueue = None     # queue for window features
        self.__plotStreamDataP = None
        self.__extractFeaturesP = None
        self.__receiveDataP = None

    def setSampleT(self, value):
        if isinstance(value, (int, float)):
            self.__sampleT = value
        else:
            print("Give int or float for sampling interval duration.")

    def setupSerialInterface(self, port, baudRate):
        self.__port = port
        self.__baudRate = baudRate

    def setFileSink(self, fileSinkName, fileSinkPath):
        self.__fileSinkName = fileSinkName
        self.__fileSinkPath = fileSinkPath

    def setBnoNormFactor(self, factors):
        print("Use with caution! Reasonable values: [2**14, 2**4, 2**4]")
        if len(factors) == 3:
            for i in range(3):
                if isinstance(factors[i], (int, float)):
                    if i == 0:
                        self.__quatNormFact = factors[i]
                    elif i == 1:
                        self.__linAccNormFact = factors[i]
                    elif i == 2:
                        self.__angVecNormFact = factors[i]
                else:
                    print("Give an int or float (!) array of length 3 with normalization values for {quat, linacc, "
                          "angvec} raw data")
        else:
            print("Give an array of length 3 with normalization values for {quat, linacc, angvec} raw data")

    def setAnaNormFact(self, factor):
        print("Use with caution! Reasonable value: 1023. When changed, check self.loadFile()!")
        if isinstance(factor, (int, float)):
            self.__anaNormFact = factor
        else:
            print("Give int or float as normalization factor for analog sensor values.")

    def startPlotStreamData(self, sensorNr, opt=None):

        def plotStreamDataF(plotSensorNr, fl):
            while True:
                data = self.__windowDataQueue.get()
                if isinstance(plotSensorNr, int):
                    if 0 <= plotSensorNr & plotSensorNr <= fl:
                        plt.ion()
                        plt.clf()
                        plt.plot(data[::, plotSensorNr])
                        plt.pause(10 ** -15)
                if isinstance(plotSensorNr, (list, tuple, np.ndarray)):
                    plt.ion()
                    plt.clf()
                    for j in range(len(plotSensorNr)):
                        plt.plot(data[::, plotSensorNr[j]])
                    plt.pause(10 ** -15)

        if self.__windowDataQueue is None:
            self.__windowDataQueue = multiprocessing.Queue()

        if self.__plotStreamDataP is None:
            self.__plotStreamDataP = multiprocessing.Process(target=plotStreamDataF, args=(sensorNr,
                                                                                           self.__frameLength, ))
            self.__plotStreamDataP.start()
            print("Started stream-data plotting process... ")
        else:
            if opt == "KILL":
                self.__plotStreamDataP.terminate()
                self.__plotStreamDataP = None
            else:
                print("PlotStreamData-process already started!")

    def selectSensorSubset(self, selectedSensors, sensorType='ir'):
        if sensorType == 'ir':
            print("...ir, currently: " + str(self.__selIrSensors))
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) <= self.__numIrSensors:
                    self.__selIrSensors = selectedSensors
                    print("...ir, update: " + str(self.__selIrSensors))
                else:
                    print(str(len(selectedSensors)) + " selected Sensors, given number of infrared sensors is " +
                          str(self.__numIrSensors) + " !")
            else:
                print("For infrared sensor selection give np.ndarray or list!")
        elif sensorType == 'fr':
            print("...fr, currently: " + str(self.__selFrSensors))
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) <= self.__numFrSensors:
                    self.__selFrSensors = selectedSensors
                    print("...fr, update: " + str(self.__selIrSensors))
                else:
                    print(str(len(selectedSensors)) + " selected Sensors, given number of infrared sensors is " +
                          str(self.__numFrSensors) + " !")
            else:
                print("For infrared sensor selection give np.ndarray or list!")
        elif sensorType == 'bno':
            print("...bno, currently: " + str(self.__selBnoData))
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) == 3:
                    for i in range(3):
                        if isinstance(selectedSensors[i], bool):
                            self.__selBnoData[i] = selectedSensors[i]
                        else:
                            print("For bno data selection give boolean array, e. g. [True, True, True] to enable all "
                                  "data (corresponding to  {quat, linacc, angvec}).")

                    print("...bno, update: " + str(self.__selIrSensors))
                else:
                    print("For bno data selection give boolean array of size 3 (!), e. g. [True, True, True] to enable "
                          "all data (corresponding to  {quat, linacc, angvec}).")
        else:
            print("sensorType has to be either \'ir\' (default), \'fr\' or \’bno\'")

    def addDataFiles(self, fileSourceName, fileSourcePath, startTime=0, stopTime=10**9, label=None):

        if isinstance(fileSourceName, str) and isinstance(fileSourcePath, str) and \
                isinstance(startTime, (int, float)) and isinstance(stopTime, (int, float)) and isinstance(label, int):
            self.fileSourceName.append(fileSourceName)
            self.fileSourcePath.append(fileSourcePath)
            self.fileSourceStartT.append(startTime)
            self.fileSourceStopT.append(stopTime)
            self.fileLabels.append(label)
        else:
            print("File source name, path, start-time, stop-time or label are incorrect data-type or format!")

    def startReceiveData(self, opt=None):

        def receiveDataF(outputQueue):

            #
            # find and connect COM port:
            #

            if self.__port is None:
                for i in range(8):
                    self.__port = "/dev/ttyUSB" + str(i)
                    try:
                        ser = serial.Serial(port=self.__port, baudrate=self.__baudRate)
                        break
                    except serial.SerialException:
                        if i == 7:
                            ser = None
                        continue
            else:
                ser = serial.Serial(port=self.__port, baudrate=self.__baudRate)
            try:
                dummy = ser.readline()
            except AttributeError:
                print("\n \n COM-port connected? Baud rate correct? Connected to another process? \n \n ")

            #
            # setup .txt-file (this eventually updates self.__fileSinkName):
            #

            dateAndTime = datetime.datetime.now()
            dateAndTime = str(dateAndTime.year) + str(dateAndTime.month) + str(dateAndTime.day) +\
                                                                        str(dateAndTime.hour) + str(dateAndTime.minute)
            if self.__fileSinkName[(len(self.__fileSinkName) - 4):] != ".txt":
                self.__fileSinkName = str(str(self.__fileSinkName) + dateAndTime + ".txt")
            outputFile = open(self.__fileSinkPath + self.__fileSinkName, "w")

            with open(file=self.__fileSinkPath + self.__fileSinkName, mode="w"):
                writer = csv.writer(outputFile, delimiter=" ")
                writer.writerow("%" + "start Time: " + str(datetime.datetime.now().hour) + "h"
                                + str(datetime.datetime.now().minute) + "m" + str(datetime.datetime.now().second) + "s"
                                + str(datetime.datetime.now().microsecond) + "us")

                #
                # format and write data:
                #

                q = [0, 0, 0, 0]                # init bno data (quaternion)
                la = [0, 0, 0]                  # angular velocity
                av = [0, 0, 0]                  # linear acceleration
                dataWindow = np.zeros([self.__windowWidth, self.__windowLength])  # initialize window
                windowCount = 0  # counting samples until next window starts

                oldLine = ser.readline()
                waitCount = 0
                while np.size(oldLine) != self.__frameLength:  # wait for complete line
                    oldLine = ser.readline()
                    oldLine = oldLine.decode("utf-8")
                    oldLine = oldLine.split(",")[:self.__frameLength]
                    if waitCount > 20:  # restart connection if stuck
                        ser.close()
                        ser.open()
                bnoData = 0
                waitCount = 0
                while bnoData == 0:  # wait for bno ready / calibration
                    oldLine = ser.readline().decode("utf-8").split(",")[:self.__frameLength]
                    if np.size(oldLine) == self.__frameLength:
                        bnoData = float(oldLine[self.__frameLength - 2]) + float(oldLine[self.__frameLength - 3]) \
                                  + float(oldLine[self.__frameLength - 4]) + float(oldLine[self.__frameLength - 5])
                    if waitCount > 20:  # restart connection if stuck
                        ser.close()
                        ser.open()
                print("Recording...")
                while True:
                    try:
                        line = ser.readline().decode("utf-8").split(",")[:self.__frameLength]
                        if np.size(line) == self.__frameLength:
                            oldLine = line
                        else:  # interpolate if message is incomplete
                            line = oldLine
                        writer.writerow(line)
                        line = np.array(line).astype(np.float)
                        if line[self.__frameLength - 1] == 0:
                            for i in range(4):
                                q[i] = line[self.__frameLength - 2 - i] / self.__quatNormFact
                        if line[self.__frameLength - 1] == 1:
                            for i in range(3):
                                la[i] = line[self.__frameLength - 3 - i] / self.__linAccNormFact
                        if line[self.__frameLength - 1] == 2:
                            for i in range(3):
                                av[i] = line[self.__frameLength - 3 - i] / self.__angVecNormFact

                        line = np.concatenate([line[:(self.__numIrSensors + self.__numFrSensors)] / self.__anaNormFact,
                                               q, la, av])
                        line[(line > 10) | (line < -10)] = 0  # threshold estimated (because of la, av)
                    except (KeyboardInterrupt, SystemExit, serial.SerialException):
                        writer.writerow("%" + "stop Time: " + str(datetime.datetime.now().hour) + "h"
                                        + str(datetime.datetime.now().minute) + "m"
                                        + str(datetime.datetime.now().second) + "s"
                                        + str(datetime.datetime.now().microsecond) + "us")
                        break
                    windowCount += 1
                    dataWindow = np.roll(dataWindow, -1, 0)
                    dataWindow[-1, ::] = line
                    if windowCount == self.__windowShift:
                        windowCount = 0
                        dataOut = np.array(dataWindow)
                        for i in range(self.__windowLength):
                            dataOut[::, i] = dataOut[::, i] * self.__windowFunction

                        # write data to queue:
                        outputQueue.put(dataOut)

        if self.__windowDataQueue is None:
            self.__windowDataQueue = multiprocessing.Queue()

        if self.__receiveDataP is None:
            self.__receiveDataP = multiprocessing.Process(target=receiveDataF, args=(self.__windowDataQueue, ))
            self.__receiveDataP.start()
            print("Started receiving process...")
        else:
            if opt == "KILL":
                self.__receiveDataP.terminate()
                self.__receiveDataP = None
            else:
                print("ReceiveData-process already started!")

    def setWindowFunction(self, functionName, alpha):

        if isinstance(alpha, (int, float)):
            self.__windowAlpha = alpha
        else:
            print("Give int or float for window shape parameter alpha.")

        if isinstance(functionName, str):
            if functionName == 'tukey':
                self.__windowFunction = scipy.signal.tukey(self.__windowWidth, self.__windowAlpha)
            elif functionName == 'rect':
                self.__windowFunction = np.ones(self.__windowWidth)
            elif functionName == 'bart':
                self.__windowFunction = np.bartlett(self.__windowWidth)
            elif functionName == 'black':
                self.__windowFunction = np.blackman(self.__windowWidth)
            elif functionName == 'ham':
                self.__windowFunction = np.hamming(self.__windowWidth)
            elif functionName == 'hann':
                self.__windowFunction = np.hanning(self.__windowWidth)
            elif functionName == 'kaiser':
                self.__windowFunction = np.kaiser(self.__windowWidth, self.__windowAlpha)
            elif functionName == 'gauss':
                self.__windowFunction = scipy.signal.gaussian(self.__windowWidth, self.__windowAlpha)
            else:
                print("Give proper function name.")
        else:
            print("Give str as window function name.")

    def loadFile(self, fileName, filePath):

        # load data from file:
        data = np.loadtxt(filePath + fileName, delimiter=',', comments="%")
        cols = np.size(data[0, ::])

        # ir data (clean transmission errors):
        irData = data[::, 0:self.__numIrSensors]
        irData[irData > self.__anaNormFact] = self.__anaNormFact
        irData[irData < 0] = 0
        irData = irData / self.__anaNormFact

        # force Data (clean transmission errors):
        forceData = self.__anaNormFact - data[::, self.__numIrSensors:self.__nrAnalogSensors]
        forceData[forceData > self.__anaNormFact] = self.__anaNormFact
        forceData[forceData < 0] = 0
        forceData = forceData / self.__anaNormFact

        # handle bno data (normalize):
        temp = data[data[::, cols - 1] == 0]  # get lines with quatData
        quatData = temp[::, self.__nrAnalogSensors:(
                self.__nrAnalogSensors + 4)] / self.__quatNormFact  # normalization (14 bit) and scale as analog data
        temp = data[data[::, cols - 1] == 1]  # get lines with lin acc data
        linAccData = temp[::, self.__nrAnalogSensors:(self.__nrAnalogSensors + 3)] / self.__linAccNormFact
        temp = data[data[::, cols - 1] == 2]  # get lines with ang vec data
        angVecData = temp[::, self.__nrAnalogSensors:(self.__nrAnalogSensors + 3)] / self.__angVecNormFact

        # quat, linaAcc and angVec possibly of different length ...:
        bnoDataSize = np.min([np.size(quatData[::, 0]), np.size(linAccData[::, 0]), np.size(angVecData[::, 0])])
        quatData = quatData[:bnoDataSize, ::]
        linAccData = linAccData[:bnoDataSize, ::]
        angVecData = angVecData[:bnoDataSize, ::]

        tAxisAnalog = np.linspace(0, self.__sampleT * (np.size(irData[::, 0]) - 1), np.size(irData[::, 0]))

        quatDataTemp = []
        linAccDataTemp = []
        angVecDataTemp = []
        for i in range(4):
            tempData = quatData[::, 3]
            tempAxis = np.linspace(0, 3 * self.__sampleT * bnoDataSize, bnoDataSize)
            dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 1), bnoDataSize + 1)
                try:
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 2), bnoDataSize + 2)
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            quatDataTemp.append(dataInterp(tAxisAnalog))
        quatData = np.array(quatDataTemp).T
        for i in range(3):
            tempData = linAccData[::, i]
            tempAxis = np.linspace(0, 3 * self.__sampleT * bnoDataSize, bnoDataSize)
            dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 1), bnoDataSize + 1)
                try:
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 2), bnoDataSize + 2)
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            linAccDataTemp.append(dataInterp(tAxisAnalog))
        linAccData = np.array(linAccDataTemp).T
        for i in range(3):
            tempData = angVecData[::, i]
            tempAxis = np.linspace(0, 3 * self.__sampleT * bnoDataSize, bnoDataSize)
            dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            try:
                quatData[i] = dataInterp(tAxisAnalog)
            except ValueError:
                tempData = np.concatenate([tempData, np.array([0])])
                tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 1), bnoDataSize + 1)
                try:
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
                except ValueError:
                    tempData = np.concatenate([tempData, np.array([0])])
                    tempAxis = np.linspace(0, 3 * self.__sampleT * (bnoDataSize + 2), bnoDataSize + 2)
                    dataInterp = scipy.interpolate.interp1d(tempAxis, tempData, kind='linear')
            angVecDataTemp.append(dataInterp(tAxisAnalog))
        angVecData = np.array(angVecDataTemp).T

        temp = []
        for i in range(self.__numIrSensors):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(irData[::, i])
            temp.append(np.transpose(temp2))
        irData = temp
        temp = []
        for i in range(self.__numFrSensors):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(forceData[::, i])
            temp.append(np.transpose(temp2))
        forceData = temp
        temp = []
        for i in range(3):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(linAccData[::, i])
            temp.append(np.transpose(temp2))
        linAccData = temp
        temp = []
        for i in range(3):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(angVecData[::, i])
            temp.append(np.transpose(temp2))
        angVecData = temp
        temp = []
        for i in range(4):
            temp2 = []
            temp2.append(tAxisAnalog)
            temp2.append(quatData[::, i])
            temp.append(np.transpose(temp2))
        quatData = temp
        return irData, forceData, quatData, linAccData, angVecData

    def plotWindowFunction(self):

        timeAxis = np.linspace(0, self.__windowWidth * self.__sampleT, self.__windowWidth)
        plt.plot(timeAxis, self.__windowFunction)
        plt.title('Time Function of ' + str(self.__windowFunction) + ' window')
        plt.xlabel('t in s')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.figure()
        freqAxis = np.linspace(0, 2 / self.__sampleT, int(self.__windowWidth))
        windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(
            np.concatenate((np.zeros(int(self.__windowWidth / 2)), self.__windowFunction, np.zeros(int(self.__windowWidth / 2)))))))
        windowFreqResponse = 20 * np.log10(windowFreqResponse / np.max(windowFreqResponse))
        plt.plot(freqAxis, windowFreqResponse[int(self.__windowWidth):])
        plt.xlim(0, )
        plt.ylim(-120, )
        plt.title("Frequency Response of chosen Window (" + str(self.__windowFunction) + ", Conv. Th.)")
        plt.xlabel('f in Hz')
        plt.ylabel('dB')
        plt.grid()
        plt.show()

    def readDataSet(self, checkData=False, equalLength=False):

        if len(self.fileSourceName) == 0:
            print("No file sources given.")
        else:
            startTimes = np.array(self.fileSourceStartT)
            stopTimes = np.array(self.fileSourceStopT)
            if equalLength:
                stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - stopTimes) + startTimes
                stopTimes = [int(x) for x in stopTimes]
                startTimes = self.fileSourceStartT

            dataSet = []
            for i, element in enumerate(self.fileSourceName):
                irData, forceData, quatData, linAccData, angVecData = self.loadFile(fileName=element,
                                                                                    filePath=self.fileSourcePath[i])
                dataSetTemp = []
                for j in range(len(self.__selIrSensors)):
                    dataSetTemp.append(irData[int(self.__selIrSensors[j])][int(startTimes[i]):int(stopTimes[i]), 1])
                for j in range(len(self.__selFrSensors)):
                    dataSetTemp.append(forceData[int(self.__selFrSensors[j])][int(startTimes[i]):int(stopTimes[i]), 1])
                if self.__selBnoData[0]:
                    for j in range(4):
                        dataSetTemp.append(quatData[j][int(startTimes[i]):int(stopTimes[i]), 1])
                if self.__selBnoData[1]:
                    for j in range(3):
                        dataSetTemp.append(linAccData[j][int(startTimes[i]):int(stopTimes[i]), 1])
                if self.__selBnoData[2]:
                    for j in range(3):
                        dataSetTemp.append(angVecData[j][int(startTimes[i]):int(stopTimes[i]), 1])
                sensorData = np.array(dataSetTemp).T
                dataSet.append(sensorData)

            if checkData:
                for i, element in enumerate(self.fileSourceName):
                    for j in range(len(dataSet[i][0, ::])):
                        plt.plot(dataSet[i][::, j], label=str(j))
                        plt.title(self.fileSourceName[i])
                    plt.legend()
                    plt.show()

            return dataSet

    def extractFeatures(self, data):

        featureVector = []
        # wavelet features:
        coeffsVal = []
        coeffsVal1 = []
        coeffsAmp = []
        coeffsAmp1 = []
        if self.__numCoeffs != 0:
            for i in range(self.__windowLength):
                data[::, i] = data[::, i]
                coeffs = pywt.wavedec(data[::, i], wavelet=self.__wavelet, mode='symmetric', level=1)
                coeffs0 = coeffs[0]
                coeffs1 = coeffs[1]
                coeffsAmp.append(coeffs0[coeffs0.argsort()[-self.__numCoeffs:]])
                coeffsAmp1.append(coeffs1[coeffs1.argsort()[-self.__numCoeffs:]])
                translationAxis = np.linspace(-1, 1, np.size(coeffs0))
                translationAxis1 = np.linspace(-1, 1, np.size(coeffs1))
                #
                if np.max(coeffs0) == 0:
                    coeffsVal.append(np.zeros(self.__numCoeffs))
                else:
                    coeffsVal.append(translationAxis[coeffs0.argsort()[-self.__numCoeffs:]])
                for j in range(np.size(coeffsVal[i])):
                    temp = coeffsVal
                    featureVector.append(temp[i][j])
                for j in range(np.size(coeffsAmp[i])):
                    temp = coeffsAmp
                    featureVector.append(temp[i][j])
                # first level
                if self.__wvltLvl1:
                    if np.max(coeffs1) == 0:
                        coeffsVal1.append(np.zeros(self.__numCoeffs))
                    else:
                        coeffsVal1.append(translationAxis1[coeffs1.argsort()[-self.__numCoeffs:]])
                    for j in range(np.size(coeffsVal1[i])):
                        temp = coeffsVal1
                        featureVector.append(temp[i][j])
                    for j in range(np.size(coeffsAmp1[i])):
                        temp = coeffsAmp1
                        featureVector.append(temp[i][j])
        # fourier features:
        freqAxis = np.linspace(-1, 1, int(self.__windowWidth))
        dominantFreqVal = []
        dominantFreqAmp = []
        dominantFreqPha = []
        if self.__numFreqs != 0:
            for i in range(self.__windowLength):
                data[::, i] = data[::, i]
                spectrum = np.fft.fftshift(np.fft.fft(data[::, i]))[int(self.__windowWidth / 2):]
                absSpectrum = np.abs(np.fft.fftshift(np.fft.fft(data[::, i])))[int(self.__windowWidth / 2):]
                reS = np.real(spectrum[absSpectrum.argsort()[-self.__numFreqs:]])
                imS = np.imag(spectrum[absSpectrum.argsort()[-self.__numFreqs:]])
                dominantFreqAmp.append(np.sqrt(reS ** 2 + imS ** 2))
                dominantFreqPha.append(np.arctan(imS / reS) + 0.01 * reS)
                dominantFreqVal.append(freqAxis[absSpectrum.argsort()[-self.__numFreqs:]])
                for j in range(np.size(dominantFreqVal[i]) - 1):
                    temp = dominantFreqVal
                    featureVector.append(temp[i][j])
                for j in range(np.size(dominantFreqAmp[i])):
                    temp = dominantFreqAmp
                    featureVector.append(temp[i][j])
                for j in range(np.size(dominantFreqPha[i])):
                    temp = dominantFreqPha
                    featureVector.append(temp[i][j])
        # statistical features
        if self.__enaStatFeats:
            xCorrWavCoeffs = 5
            for i in range(self.__windowLength):
                featureVector.append(np.mean(data[::, i]))
                featureVector.append(np.var(data[::, i]))
                featureVector.append(mad(data[::, i]))
                for j in range(self.__windowLength - i - 1):
                    correlation = np.correlate(data[::, i], data[::, j + 1], mode='same') \
                                  / np.sum(data[::, i]) / np.size(data[::, i])
                    coeffs = pywt.wavedec(correlation, wavelet=self.__wavelet, mode='symmetric', level=0)
                    coeffs0 = coeffs[0]
                    translationAxis = np.linspace(-1, 1, np.size(coeffs0))
                    domCorrCoeffsAmp = coeffs0[coeffs0.argsort()[-xCorrWavCoeffs:]]
                    if np.max(coeffs0) == 0:
                        domCorrCoeffsVal = np.zeros(xCorrWavCoeffs)
                    else:
                        domCorrCoeffsVal = translationAxis[coeffs0.argsort()[-xCorrWavCoeffs:]]
                    for k in range(xCorrWavCoeffs):
                        featureVector.append(domCorrCoeffsVal[k])
                    for k in range(xCorrWavCoeffs):
                        featureVector.append(domCorrCoeffsAmp[k])
        featureVector = np.reshape(featureVector, np.size(featureVector))
        return featureVector

    def initFeatureQueue(self, opt=None):

        def extractFeaturesF(inputQueue, outputQueue):
            while True:
                currentWindow = inputQueue.get()
                features = self.extractFeatures(data=currentWindow)
                outputQueue.put(features)

        if self.__windowDataQueue is None:
            self.__windowDataQueue = multiprocessing.Queue()

        if self.__featureQueue in None:
            self.__featureQueue = multiprocessing.Queue()

        if self.__extractFeaturesP is None:
            self.__extractFeaturesP = multiprocessing.Process(target=extractFeaturesF, args=(self.__windowDataQueue,
                                                                                             self.__featureQueue))
            self.__extractFeaturesP.start()
            print("Started feature extraction process...")
        else:
            if opt == "KILL":
                self.__extractFeaturesP.terminate()
                self.__extractFeaturesP = None
            else:
                print("Feature extraction-process already started!")

    def featureNormalization(self, features, initTrue=True):
        if not initTrue:
            if np.size(features) == len(features):
                print("Single feature vector for normalization!?")
            if self.__featNormMethod == 'stand':
                mue = []
                sigma = []
                for i in range(len(features[0][::])):
                    mue.append(np.mean(features[::, i]))
                    sigma.append(np.std(features[::, i]))
                    if sigma[i] == 0:
                        sigma[i] = np.mean(features[0, i]) * 10 ** 6 + 10 ** 6
                    features[::, i] = (features[::, i] - mue[i]) / sigma[i]
                self.__normVal = np.array([mue, sigma])
            elif self.__featNormMethod == 'mean':
                mue = []
                minVal = []
                maxVal = []
                for i in range(len(features[0][::])):
                    mue.append(np.mean(features[::, i]))
                    minVal.append(np.min(features[::, i]))
                    maxVal.append(np.max(features[::, i]))
                    if (maxVal[i] - minVal[i]) == 0:
                        maxVal[i] = np.abs(minVal[i] * 10 ** 6)
                    features[::, i] = (features[::, i] - mue[i]) / (maxVal[i] - minVal[i])
                self.__normVal = np.array([mue, minVal, maxVal])
            elif self.__featNormMethod == 'minmax':
                minVal = []
                maxVal = []
                for i in range(len(features[0][::])):
                    minVal.append(np.min(features[::, i]))
                    maxVal.append(np.max(features[::, i]))
                    if (maxVal[i] - minVal[i]) == 0:
                        maxVal[i] = np.abs(minVal[i] * 10 ** 6)
                    features[::, i] = (features[::, i] - minVal[i]) / (maxVal[i] - minVal[i])
                self.__normVal = np.array([minVal, maxVal])
            else:
                print("Give proper str for feature normalization method (stand, minmax or mean)!")
            return features
        else:
            if self.__normVal is None:
                print("Feature normalization not initialized yet! "
                      "-> use self.initFeatureNormalization(trainingDataSet).")
            else:
                if self.__featNormMethod == 'stand':
                    mue = self.__normVal[::, 0]
                    sigma = self.__normVal[::, 1]
                    features = (features - mue) / sigma
                elif self.__featNormMethod == 'mean':
                    mue = self.__normVal[::, 0]
                    minVal = self.__normVal[::, 1]
                    maxVal = self.__normVal[::, 2]
                    features = (features - mue) / (maxVal - minVal)
                elif self.__featNormMethod == 'minmax':
                    minVal = self.__normVal[::, 0]
                    maxVal = self.__normVal[::, 1]
                    features = (features - minVal) / (maxVal - minVal)
                else:
                    print("Give proper str for feature normalization method (stand, minmax or mean)!")
                return features

    def windowSourceData(self, dataSet, enaWindowC=False):

        if len(dataSet) != len(self.fileLabels):
            print("Dimension mismatch: labels < - > dataSet")

        windowedData = []
        windowLabels = []
        for k in range(len(dataSet)):

            data = dataSet[k]

            minDataLength = []
            for i in range(self.__windowLength):
                minDataLength.append(np.size(data[::, i]))
            minDataLength = np.min(minDataLength)
            tempNumWindows = int((minDataLength - self.__windowWidth) / self.__windowShift + 1)
            for i in range(tempNumWindows):
                windowedDataTemp = []
                for j in range(self.__windowLength):
                    dataTemp = data[i * self.__windowShift:(i * self.__windowShift + self.__windowWidth), j]
                    windowedDataTemp.append(dataTemp * self.__windowFunction)
                windowedDataTemp = np.transpose(windowedDataTemp)
                windowedData.append(windowedDataTemp)
                windowLabels.append(self.fileLabels[k])
                if enaWindowC:
                    plt.plot(windowedDataTemp)
                    plt.title("Label:=" + str(self.fileLabels[k]))
                    plt.show()
        return windowedData, windowLabels

    def splitTrainTest(self):
        pass

    def initFeatNormalization(self, dataSet):

        dataWindows, windowLabels = self.windowSourceData(dataSet)
        windowFeatures = []
        for i in range(len(dataWindows)):
            windowFeatures.append(self.extractFeatures(dataWindows[i]))

        windowFeatures = self.featureNormalization(features=windowFeatures, initTrue=False)



if __name__ == '__main__':

    a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numFreqs=10, numCoeffs=10,
                          enaStatFeats=True)

    a.setFileSink(fileSinkName="test.txt", fileSinkPath="../")

    a.startPlotStreamData(sensorNr=0)

    a.startReceiveData()




