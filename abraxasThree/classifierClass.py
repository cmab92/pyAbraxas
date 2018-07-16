# -*- coding: utf-8 -*-
"""
cb, 11.07.2018

 - same same but classy

"""
import serial
import datetime
import csv
import pywt
import random
import scipy.signal
import scipy.interpolate
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust import mad
from sklearn import svm
from six.moves import cPickle


class AbraxasClassifier:

    def __init__(self, numIrSensors, numFrSensors, windowWidth, windowShift, numCoeffs, numFreqs, kernel,
                 enaStatFeats=True, wavelet='haar', wvltLvl1=False, featNormMethod='stand', trainFraction=0.66,
                 classSortTT=True, randomSortTT=False, lineThresholdAfterNorm=10):

        # Hardware information and parameters
        self.__numIrSensors = numIrSensors
        self.__numFrSensors = numFrSensors
        self.__nrAnalogSensors = self.__numIrSensors + self.__numFrSensors
        self.__frameLength = self.__nrAnalogSensors + 5         # 5 for bno data including index (received format)
        self.__numOfSensorsAvailable = self.__nrAnalogSensors + 10  # 10 for bno
        self.__port = None                                      # see function receive data
        self.__baudRate = 57600
        self.__sampleT = 0.0165

        # File sources and sinks
        self.__fileSinkName = ""
        self.__fileSinkPath = "../"
        self.__fileSourceName = []
        self.__fileSourcePath = []
        self.__fileSourceStartT = []
        self.__fileSourceStopT = []

        self.__fileLabels = []

        # Sensor selection
        self.__selIrSensors = np.linspace(0, self.__numIrSensors - 1, self.__numIrSensors, dtype=int)  # select all ir
        self.__selFrSensors = np.linspace(0, self.__numFrSensors - 1, self.__numFrSensors, dtype=int)  # select all fr
        self.__selBnoData = [True, True, True]   # select all bno data {quat, linacc, angvec}
        self.__numIrUsed = numIrSensors
        self.__numFrUsed = numFrSensors
        self.__numBnoUsed = 10
        self.__numOfSensorsUsed = self.__numFrUsed + self.__numIrUsed + self.__numBnoUsed
        self.__indexSensorsUsed = np.linspace(0, self.__numOfSensorsUsed - 1, self.__numOfSensorsUsed, dtype=int)  # ...
        # ... index referring to the original data-frame from serial interface

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
        self.__lineThresholdAfterNorm = lineThresholdAfterNorm  # not checked
        self.__normVal = None
        self.__trainFraction = trainFraction
        self.__kernel = kernel
        self.__numberOfClasses = None
        self.__numberWindowPerClass = None
        self.__classSortTT = classSortTT
        self.__randomSortTT = randomSortTT

        # Queues and processes
        self.__windowDataQueue = None     # queue for window data
        self.__featureQueue = None     # queue for window features
        self.__plotStreamDataP = None
        self.__extractFeaturesP = None
        self.__receiveDataP = None
        self.__liveClassificationP = None

        # Data
        self.__sourceData = None
        self.__sourceTrainWindowData = None
        self.__sourceTestWindowData = None
        self.__sourceTrainLabel = None
        self.__sourceTestLabel = None
        self.__sourceTrainFeat = None
        self.__className = []

        # Classifier
        self.__trainedClassifier = None

    def setSampleT(self, value):

        """
        ...
        :param value: sampleT does affect time and frequency axis only.
        :return: writes to:
         - self.__sampleT
        """

        if isinstance(value, (int, float)):
            self.__sampleT = value
        else:
            print("(setSampleT) Give int or float for sampling interval duration.")

    def setupSerialInterface(self, port, baudRate):

        """
        ...
        :param port: serial port Name
        :param baudRate: baud rate in baud/s
        :return: writes to:
         - self.__port
         - self.__baudRate
        """

        if isinstance(port, str):
            self.__port = port
        elif isinstance(port, int):
            self.__port = "/dev/ttyUSB" + str(port)
        else:
            print("(setupSerialInterface) Give str or port number (e. g. \'/dev/ttyUSB0\' or 0).")

        if isinstance(baudRate, int):
            self.__baudRate = baudRate
        else:
            print("(setupSerialInterface) Give int as baudrate.")

    def setFileSink(self, fileSinkName, fileSinkPath):

        """
        ...
        :param fileSinkName: Name of the sink file, to which streamed data is stored. The file is named by yymmddhhmm,
        if not given (""). File name is completed with yymmddhhmm, if the input string is not ended by .txt. Default
        is "".
        :param fileSinkPath: Path to .txt-file, in which data is stored (if not given, data is stored to the current
        working parent directory). Default "/..".
        :return: writes to:
         - self.__fileSinkName
         - self.__fileSinkPath
        """

        if isinstance(fileSinkName, str):
            self.__fileSinkName = fileSinkName
        else:
            print("(setFileSink) Give a string as file sink name.")
        if isinstance(fileSinkPath, str):
            self.__fileSinkPath = fileSinkPath
        else:
            print("(setFileSink) Give a string as file sink path.")

    def setBnoNormFactor(self, factors):

        """
        ...
        :param factors: Normalization factors for bno data. 1 quaternion is 2**14 LSB, hence default is 2**14. Factors
        for linear acceleration and angular velocity are chosen empirically (default 2**4).
        :return: writes to:
         - self.__quatNormFact
         - self.__linAccNormFact
         - self.__angVecNormFact
        """

        print("(setBnoNormFactor) Use with caution! Reasonable values: [2**14, 2**4, 2**4]")
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
                    print("(setBnoNormFactor) Give an int or float (!) array of length 3 with normalization values for "
                          "{quat, linacc, angvec} raw data")
        else:
            print("(setBnoNormFactor) Give an array of length 3 with normalization values for {quat, linacc, angvec} "
                  "raw data")

    def setAnaNormFact(self, factor):

        """
        ...
        :param factor:  Normalization factors for analog sensor data. Due to 10bit adc on atmega 2560: default 1023.
        :return: -
        """

        print("(setAnaNormFact) Use with caution! Reasonable value: 1023. When changed, check self.loadFile()!")
        if isinstance(factor, (int, float)):
            self.__anaNormFact = factor
        else:
            print("(setAnaNormFact) Give int or float as normalization factor for analog sensor values.")

    def selectSensorSubset(self, selectedSensors, sensorType):

        """
        Allows to choose a subset of variables/sensors, which is used fpr classification.
        :param selectedSensors: Give a int (analog sensors) or bool (bno) array indicating which sensors of the given
        type are to be used. For bno give an array [bool, bool, bool] indicating whether {quat, lincacc, angvec} is to
        be used.
        :param sensorType: Specifying which sensors are selected by selectedSensors ('ir', 'fr' or 'bno').
        :return: writes to:
         - self.__selIrSensors
         - self.__numIrUsed
         - self.__selFrSensors
         - self.__numFrUsed
         - self.__selBnoData
         - self.__numBnoUsed
         - self.__numOfSensorsUsed
         - self.__indexSensorsUsed
        """

        if sensorType == 'ir':
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) <= self.__numIrSensors:
                    self.__selIrSensors = selectedSensors
                    self.__numIrUsed = len(selectedSensors)
                    print("(selectSensorSubset) ...ir, used sensors: " + str(self.__selIrSensors))
                else:
                    print("(selectSensorSubset) " + str(len(selectedSensors)) + " selected Sensors, given number of "
                                                                                "infrared sensors is " +
                          str(self.__numIrSensors) + " !")
            else:
                print("(selectSensorSubset) For infrared sensor selection give np.ndarray or list!")
        elif sensorType == 'fr':
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) <= self.__numFrSensors:
                    self.__selFrSensors = selectedSensors
                    self.__numFrUsed = len(selectedSensors)
                    print("(selectSensorSubset) ...fr, used sensors: " + str(self.__selFrSensors))
                else:
                    print("(selectSensorSubset) " + str(len(selectedSensors)) + " selected Sensors, given number of "
                                                                                "infrared sensors is " +
                          str(self.__numFrSensors) + " !")
            else:
                print("(selectSensorSubset) For force sensor selection give np.ndarray or list!")
        elif sensorType == 'bno':
            self.__numBnoUsed = 0
            if isinstance(selectedSensors, (list, np.ndarray)):
                if len(selectedSensors) == 3:
                    for i in range(3):
                        if isinstance(selectedSensors[i], bool):
                            self.__selBnoData[i] = selectedSensors[i]
                            if selectedSensors[i]:
                                if i == 0:
                                    self.__numBnoUsed += 4
                                else:
                                    self.__numBnoUsed += 3
                        else:
                            print("(selectSensorSubset) For bno data selection give boolean array, e. g. [True, True, "
                                  "True] to enable all "
                                  "data (corresponding to  {quat, linacc, angvec}).")

                    print("(selectSensorSubset) ...bno, used sensors: " + str(self.__selBnoData))
                else:
                    print("(selectSensorSubset) For bno data selection give boolean array of size 3 (!), e. g. [True, "
                          "True, True] to enable "
                          "all data (corresponding to  {quat, linacc, angvec}).")
        else:
            print("(selectSensorSubset) ensorType has to be either \'ir\' (default), \'fr\' or \’bno\'")

        self.__numOfSensorsUsed = self.__numIrUsed + self.__numFrUsed + self.__numBnoUsed

        self.__indexSensorsUsed = []
        for i in range(self.__numIrUsed):
            self.__indexSensorsUsed.append(self.__selIrSensors[i])
        for i in range(self.__numFrUsed):
            self.__indexSensorsUsed.append(self.__selFrSensors[i])
        for i in range(3):
            if self.__selBnoData[i]:
                if i == 0:
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 1)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 2)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 3)
                elif i == 1:
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 4)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 5)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 6)
                else:
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 7)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 8)
                    self.__indexSensorsUsed.append(self.__nrAnalogSensors + 9)

    def addDataFiles(self, fileSourceName, fileSourcePath, startTime=0, stopTime=10**9, label=None, className=None):

        """
        Add a source file to the dataset. Given inputs are stored in arrays.
        :param fileSourceName: String specifying the file name. Default "".
        :param fileSourcePath: String specifying the file path. Default "../".
        :param startTime: Time (in samples) from which the data in the file is read.
        :param stopTime:Time (in samples) until which the data in the file is read.
        :param label: Label of the file.
        :param className: Class name displayed by some functions (self.__liveClassification).
        :return: writes to:
         - self.__fileSourceName
         - self.__fileSourcePath
         - self.__fileSourceStartT
         - self.__fileSourceStopT
         - self.__fileLabels
         - self.__numberOfClasses
         - self.__className
         - self.__numberWindowPerClass (init)
        """

        if isinstance(fileSourceName, str) and isinstance(fileSourcePath, str) and \
                isinstance(startTime, (int, float)) and isinstance(stopTime, (int, float)) and isinstance(label, int):
            self.__fileSourceName.append(fileSourceName)
            self.__fileSourcePath.append(fileSourcePath)
            self.__fileSourceStartT.append(startTime)
            self.__fileSourceStopT.append(stopTime)
            self.__fileLabels.append(label)
        else:
            print("(addDataFiles) File source name, path, start-time, stop-time or label are incorrect data-type or "
                  "format!")

        self.__numberOfClasses = len(set(self.__fileLabels))
        self.__numberWindowPerClass = np.zeros(self.__numberOfClasses)

        newClass = True
        for i in range(self.__numberOfClasses - 1):
            if label == self.__fileLabels[i]:
                newClass = False

        if newClass is True:
            if isinstance(className, str):
                self.__className.append(className)
            else:
                self.__className.append(self.__fileSourceName[:-4])

    def setWindowFunction(self, functionName, alpha):

        """
        
        :param functionName: Chooses a window function of the following (https://en.wikipedia.org/wiki/Window_function):
        tukey -> tukey window (flattened cosine)
        rect -> rectangular window
        bart -> bartlett window
        black -> blackman window
        ham -> hamming window
        hann -> hanning window (raised-cosine window)
        kaiser -> kaiser window
        gauss -> gaussian window
        Default "tukey".
        :param alpha: Shape parameter of window function (not relevant for all). Default 0.1.
        :return: writes to:
         - self.__windowFunction
         - self.__windowAlpha
        """

        if isinstance(alpha, (int, float)):
            self.__windowAlpha = alpha
        else:
            print("(setWindowFunction) Give int or float for window shape parameter alpha.")

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
                print("(setWindowFunction) Give proper function name.")
        else:
            print("(setWindowFunction) Give str as window function name.")

    def loadFile(self, fileName, filePath):

        """
        Load data from single .txt-file, ensure 0 < analogData < 1023 (1), split data to infrared, force, quaternion,
        linear acceleration and angular velocity data. Also interpolate bno data linearly (same axis as analog data).
        Consider self.__angVecNormFact, self,__linAccNormFact, self.__quatNormFact, self.__anaNormFact!
        :param fileName:
        :param filePath:
        :return: irData, forceData, quatData, linAccData and angVecData. Each as an np.ndarray with time axis.
        """

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

        """
        Simply plot time function and spectrum of window function.
        :return: -
        """

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

        """
        Read the data-set specified by addDataFiles. Triggers self.__sourceData!
        :param checkData: If True the data-set is plotted.
        :param equalLength: Is True the data read from different file is cut to equal length.
        :return: Array with shape dataSet[i][j, k], where i refers to the i-th file loaded, k indicates the sensor and
        j is the "time"-index.
        Writes to:
         - self.__sourceData
        """

        if len(self.__fileSourceName) == 0:
            print("(readDataSet) No file sources given.")
        else:
            startTimes = np.array(self.__fileSourceStartT)
            stopTimes = np.array(self.__fileSourceStopT)
            if equalLength:
                stopTimes = np.ones([len(startTimes)]) * np.min(stopTimes - startTimes) + startTimes
                stopTimes = [int(x) for x in stopTimes]
                startTimes = self.__fileSourceStartT
            dataSet = []
            for i, element in enumerate(self.__fileSourceName):
                irData, forceData, quatData, linAccData, angVecData = self.loadFile(fileName=element,
                                                                                    filePath=self.__fileSourcePath[i])
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
                for i, element in enumerate(self.__fileSourceName):
                    for j in range(len(dataSet[i][0, ::])):
                        plt.plot(dataSet[i][::, j], label=str(j))
                        plt.title(self.__fileSourceName[i])
                    plt.legend()
                    plt.show()

            self.__sourceData = np.array(dataSet)

            self.windowSplitSourceDataTT()

            return np.array(dataSet)

    def extractFeatures(self, data):

        """
        Extracts features of a single data window.
        :param data: Input data, given as data[j,k], with j-th data-point, k-th sensor.
        :return: A single feature vector.
        """

        featureVector = []
        np.seterr(all='raise')

        # wavelet features:

        coeffsVal = []
        coeffsVal1 = []
        coeffsAmp = []
        coeffsAmp1 = []
        if self.__numCoeffs != 0:
            for i in range(self.__numOfSensorsUsed):
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
            for i in range(self.__numOfSensorsUsed):
                data[::, i] = data[::, i]
                spectrum = np.fft.fftshift(np.fft.fft(data[::, i]))[int(self.__windowWidth / 2):]
                absSpectrum = np.abs(np.fft.fftshift(np.fft.fft(data[::, i])))[int(self.__windowWidth / 2):]
                reS = np.real(spectrum[absSpectrum.argsort()[-self.__numFreqs:]])
                imS = np.imag(spectrum[absSpectrum.argsort()[-self.__numFreqs:]])
                dominantFreqAmp.append(np.sqrt(reS ** 2 + imS ** 2))
                try:
                    dominantFreqPha.append(np.arctan(imS / reS))
                except FloatingPointError:
                    temp = np.arctan(imS / reS)
                    temp[reS == 0] = 0
                    dominantFreqPha.append(temp)
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
            for i in range(self.__numOfSensorsUsed):
                featureVector.append(np.mean(data[::, i]))
                featureVector.append(np.var(data[::, i]))
                featureVector.append(mad(data[::, i]))
                for j in range(self.__numOfSensorsUsed - i - 1):
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

        """
        Data queue, processing feature data for live classification. Initialization by classifier.
        :param opt:  If "KILL" ends process. Default None.
        :return: writes to:
         - self.__windowDataQueue
         - self.__featureQueue
         - self.__extractFeaturesP
        """

        def extractFeaturesF(inputQueue, outputQueue):
            while True:
                currentWindow = inputQueue.get()
                features = self.extractFeatures(data=currentWindow)
                features = self.featureNormalization(features=features)
                outputQueue.put(features)

        if self.__windowDataQueue is None:
            self.__windowDataQueue = multiprocessing.Queue()

        if self.__featureQueue is None:
            self.__featureQueue = multiprocessing.Queue()

        if self.__extractFeaturesP is None:
            self.__extractFeaturesP = multiprocessing.Process(target=extractFeaturesF, args=(self.__windowDataQueue,
                                                                                             self.__featureQueue))
            self.__extractFeaturesP.start()
            print("(initFeatureQueue) Started feature extraction process...")
        else:
            if opt == "KILL":
                self.__extractFeaturesP.terminate()
                self.__extractFeaturesP = None
            else:
                print("(initFeatureQueue) Feature extraction-process already started!")

    def featureNormalization(self, features, initDone=True):

        """
        Finds parameters for feature normalization and performs it. Features causing floating point error during
        initialization are suppressed by a factor of 100.
        :param features: Input features. Array of vector.
        :param initDone: If False, the input data is used to determine the normalization parameters.
        :return: Normalized features. Array or vector.
        Writes to:
         - self.__normVal
        """

        np.seterr(all='raise')

        if not initDone:

            features = np.array(features)

            if np.size(features) == len(features):
                print("(featureNormalization) Single feature vector for normalization!?")
            if self.__featNormMethod == 'stand':
                mue = []
                sigma = []
                for i in range(len(features[0][::])):
                    mue.append(np.mean(features[::, i]))
                    sigma.append(np.std(features[::, i]))
                    if sigma[i] == 0:
                        sigma[i] = np.mean(features[0, i]) * 10 ** 6 + 10 ** 6
                    try:
                        features[::, i] = (features[::, i] - mue[i]) / sigma[i]
                    except FloatingPointError:
                        sigma[i] = 1
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
                        maxVal[i] = np.abs(minVal[i] + 1)
                    try:
                        features[::, i] = (features[::, i] - mue[i]) / (maxVal[i] - minVal[i])
                    except FloatingPointError:
                        maxVal[i] = minVal[i] + 1
                        features[::, i] = (features[::, i] - mue[i]) / (maxVal[i] - minVal[i])
                self.__normVal = np.array([mue, minVal, maxVal])
            elif self.__featNormMethod == 'minmax':
                minVal = []
                maxVal = []
                for i in range(len(features[0][::])):
                    minVal.append(np.min(features[::, i]))
                    maxVal.append(np.max(features[::, i]))
                    if (maxVal[i] - minVal[i]) == 0:
                        maxVal[i] = np.abs(minVal[i] + 1)
                    try:
                        features[::, i] = (features[::, i] - minVal[i]) / (maxVal[i] - minVal[i])
                    except FloatingPointError:
                        maxVal[i] = minVal[i] + 1
                        features[::, i] = (features[::, i] - minVal[i]) / (maxVal[i] - minVal[i])
                self.__normVal = np.array([minVal, maxVal])
            else:
                print("(featureNormalization)Give proper str for feature normalization method (stand, minmax or mean)!")

            return features
        else:

            if self.__normVal is None:
                print("(featureNormalization) Feature normalization not initialized yet! "
                      "-> use self.initFeatureNormalization(trainingDataSet).")
            else:
                if self.__featNormMethod == 'stand':
                    mue = self.__normVal[0, ::]
                    sigma = self.__normVal[1, ::]
                    try:
                        features = (features - mue) / sigma
                    except FloatingPointError:
                        for i in range(len(features)):
                            try:
                                features[i] = (features[i] - mue[i]) / sigma[i]
                            except FloatingPointError:
                                features[i] = 0
                elif self.__featNormMethod == 'mean':
                    mue = self.__normVal[0, ::]
                    minVal = self.__normVal[1, ::]
                    maxVal = self.__normVal[2, ::]
                    try:
                        features = (features - mue) / (maxVal - minVal)
                    except FloatingPointError:
                        for i in range(len(features)):
                            try:
                                features[i] = (features[i] - mue[i]) / (maxVal[i] - minVal[i])
                            except FloatingPointError:
                                features[i] = 0
                elif self.__featNormMethod == 'minmax':
                    minVal = self.__normVal[0, ::]
                    maxVal = self.__normVal[1, ::]
                    try:
                        features = (features - minVal) / (maxVal - minVal)
                    except FloatingPointError:
                        for i in range(len(features)):
                            try:
                                features[i] = (features[i] - minVal[i]) / (maxVal[i] - minVal[i])
                            except FloatingPointError:
                                features[i] = 0
                else:
                    print("(featureNormalization)Give proper str for feature normalization method (stand, minmax or "
                          "mean)!")
                return features

    def windowSplitSourceDataTT(self, inputData=None, enaWindowC=False):

        """
        Split source data-set to test and training data.
        :param inputData: If None the source data is used.
        :param enaWindowC: If True plots data windows.
        :return: The whole windowed data-set with labels as windowedData, windowLabels.
        Writes to:
         - self.__numberWindowPerClass
         - self.__sourceTrainWindowData = trainingData
         - self.__sourceTrainLabel = trainingLabels
         - self.__sourceTestWindowData = testData
         - self.__sourceTestLabel = testLabels
        """

        if inputData is None:
            inputData = self.__sourceData
            if self.__sourceData is None:
                print("(windowSplitSourceDataTT) No source data specified yet.")
                return False

        if len(inputData) != len(self.__fileLabels):
            print("(windowSplitSourceDataTT) Dimension mismatch: labels < - > dataSet")

        windowedData = []
        windowLabels = []
        for k in range(len(inputData)):

            data = inputData[k]

            minDataLength = []
            for i in range(self.__numOfSensorsUsed):
                minDataLength.append(np.size(data[::, i]))
            minDataLength = np.min(minDataLength)
            tempNumWindows = int((minDataLength - self.__windowWidth) / self.__windowShift + 1)
            for i in range(tempNumWindows):
                windowedDataTemp = []
                for j in range(self.__numOfSensorsUsed):
                    dataTemp = data[i * self.__windowShift:(i * self.__windowShift + self.__windowWidth), j]
                    windowedDataTemp.append(dataTemp * self.__windowFunction)
                windowedDataTemp = np.transpose(windowedDataTemp)
                windowedData.append(windowedDataTemp)
                windowLabels.append(self.__fileLabels[k])
                if enaWindowC:
                    plt.plot(windowedDataTemp)
                    plt.title("Label:=" + str(self.__fileLabels[k]))
                    plt.show()

        testData = []
        trainingData = []
        testLabels = []
        trainingLabels = []

        for i in range(len(windowedData)):
            self.__numberWindowPerClass[windowLabels[i]] += 1

        classCount = np.zeros(np.shape(self.__numberWindowPerClass))

        if self.__classSortTT and not self.__randomSortTT:
            for i in range(len(windowedData)):
                if ((classCount[windowLabels[i]])/(self.__numberWindowPerClass[windowLabels[i]] - 1)) < \
                        self.__trainFraction:
                    trainingData.append(windowedData[i])
                    trainingLabels.append(windowLabels[i])
                    classCount[windowLabels[i]] += 1
                else:
                    testData.append(windowedData[i])
                    testLabels.append(windowLabels[i])
        elif self.__randomSortTT and not self.__classSortTT:
            index = np.linspace(0, len(windowedData)-1, len(windowedData), dtype=int)
            random.shuffle(index)
            for i in range(len(windowedData)):
                if i/(len(windowedData) - 2) < self.__trainFraction:
                    trainingData.append(windowedData[index[i]])
                    trainingLabels.append(windowLabels[index[i]])
                else:
                    testData.append(windowedData[index[i]])
                    testLabels.append(windowLabels[index[i]])
        else:
            print("(windowSplitSourceDataTT) Either classSortTT or randomSortTT has to be True (classSort gives "
                  "trainFraction of each class as training data; randomSort takes arbitrary samples, hence classes "
                  "may not be trained...). ")

        self.__sourceTrainWindowData = trainingData
        self.__sourceTrainLabel = trainingLabels
        self.__sourceTestWindowData = testData
        self.__sourceTestLabel = testLabels

        print("(windowSplitSourceDataTT) Input or source data is windowed and split to test and training data "
              "( ratio: ", str(self.__trainFraction), "). RandomSort is " + str(self.__randomSortTT) + ", classSort is "
              + str(self.__classSortTT) + ".")
        return windowedData, windowLabels

    def trainClassifier(self, classifier=None, oneClass=False):

        """
        Train classifier with source training data.
        :param classifier: If None initializes svc.
        :return: writes to:
         - self.__trainedClassifier
        """

        if classifier is None:
            classifier = svm.SVC(kernel=self.__kernel)

        if oneClass is False:
            classifier.fit(self.__sourceTrainFeat, self.__sourceTrainLabel)
        else:
            classifier.fit(self.__sourceTrainFeat)

        self.__trainedClassifier = classifier

    def testClassifier(self, inputData=None, classifier=None):

        """
        Test classifier and plot confusion matrix.
        :param inputData: If None uses
        :param classifier: If None uses self.__trainedClassifier.
        :return: -
        """

        np.seterr(all='raise')

        if classifier is None:
            classifier = self.__trainedClassifier
            if self.__trainedClassifier is None:
                print("(testClassifier) No classifier trained yet!")
                return False

        if inputData is None:
            inputData = self.__sourceTestWindowData

        occurrenceCount = np.zeros(self.__numberOfClasses)
        confMat = np.zeros([self.__numberOfClasses, self.__numberOfClasses])

        for i in range(len(inputData)):
            normedFeatVec = self.featureNormalization(self.extractFeatures(inputData[i]))
            prediction = classifier.predict(normedFeatVec.reshape(1, -1))
            occurrenceCount[int(self.__sourceTestLabel[i])] += 1
            confMat[int(prediction), int(self.__sourceTestLabel[i])] += 1

        for i in range(self.__numberOfClasses):
            try:
                confMat[::, i] = confMat[::, i] / occurrenceCount[i]
            except FloatingPointError:
                print("(testClassifier) Seems like class " + str(i) + " was not trained. Consider classSort as True or "
                                                                      "smaller fraction of training data "
                                                                      "(self.__trainFraction).")
            print("(testClassifier) For class " + str(i) + " the number of test samples/windows is "
                  + str(occurrenceCount[i]))

        print("(testClassifier) The overall error is " + str(100 - np.sum(np.diag(confMat))/self.__numberOfClasses*100)
              + "%.")

        self.plotMatrixWithValues(confMat)

    def initFeatNormalization(self, inputData=None, dumpName=None):

        """
        Initialize feature normalization and dumps normalization parameters.
        :param inputData: If None using source training data.
        :param dumpName: Name of parameter dump.
        :return: writes to:
         - self.__sourceTrainFeat
        """

        if inputData is None:
            inputData = self.__sourceTrainWindowData
            if self.__sourceTrainWindowData is None:
                print("(initFeatNormalization) No training data specified yet.")
                return False

        if isinstance(dumpName, str):
            if str(dumpName[(len(dumpName) - 4):]) == ".pkl":
                dumpName = dumpName
            else:
                dumpName = dumpName + ".pkl"
        else:
            dumpName = "normValDump.pkl"

        windowFeatures = []
        for i in range(len(inputData)):
            print("(initFeatNormalization) Progress feature extraction: " + str(100*i/(len(inputData))) + "%")
            windowFeatures.append(self.extractFeatures(inputData[i]))

        self.__sourceTrainFeat = self.featureNormalization(features=windowFeatures, initDone=False)

        with open(dumpName, 'wb') as normValDump:
            cPickle.dump(self.__normVal, normValDump)

    def dumpClassifier(self, dumpName=None, classifier=None):

        """
        Dumps (trained) classifier.
        :param dumpName: Give name for dump. Default "classifierDump.pkl".
        :param classifier: If None uses self.__trainedClassifier.
        :return: -
        """

        if isinstance(dumpName, str):
            if str(dumpName[(len(dumpName) - 4):]) == ".pkl":
                dumpName = dumpName
            else:
                dumpName = dumpName + ".pkl"
        else:
            dumpName = "classifierDump.pkl"

        if classifier is None:
            classifier = self.__trainedClassifier
            if self.__trainedClassifier is None:
                print("(dumpClassifier) No classifier trained yet!")

        with open(dumpName, 'wb') as classifierDump:
            cPickle.dump(classifier, classifierDump)

    def loadDumpClassifier(self, dumpName=None):

        """
        Load dumped Classifier.
        :param dumpName: Default "classifierDump.pkl".
        :return: writes to:
         - self.__trainedClassifier
        """

        if isinstance(dumpName, str):
            if str(dumpName[(len(dumpName) - 4):]) == ".pkl":
                dumpName = dumpName
            else:
                dumpName = dumpName + ".pkl"
        else:
            dumpName = "classifierDump.pkl"

        with open(dumpName, 'rb') as classifierDump:
            self.__trainedClassifier = cPickle.load(classifierDump)

    def loadDumpNormParam(self, dumpName=None):

        """
        Load dumped normalization parameters.
        :param dumpName: Default "normValDump.pkl".
        :return: writes to:
         - self.__normVal
        """

        if isinstance(dumpName, str):
            if str(dumpName[(len(dumpName) - 4):]) == ".pkl":
                dumpName = dumpName
            else:
                dumpName = dumpName + ".pkl"
        else:
            dumpName = "normValDump.pkl"

        with open(dumpName, 'rb') as normValDump:
            self.__normVal = cPickle.load(normValDump)

    def startReceiveData(self, opt=None):

        """
        Start process to receive serial data via com port. Write data to file and to multiprocessing.Queue.
        :param opt: If "KILL" ends process. Default None.
        :return: writes to:
         - self.__port
         - self.__fileSinkName
         - self.__windowDataQueue
         - self.__receiveDataP
         - self.__receiveDataP
        """

        def receiveDataF(outputQueue):

            # find and connect COM port:

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
                print("\n \n (startReceiveData) No serial interface!! COM-port connected? Baud rate correct? Connected "
                      "to another process? "
                      "\n \n ")
                return False

            # setup .txt-file (this eventually updates self.__fileSinkName):

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

                # format and write data:

                q = [0, 0, 0, 0]                # init bno data (quaternion)
                la = [0, 0, 0]                  # angular velocity
                av = [0, 0, 0]                  # linear acceleration
                dataWindow = np.zeros([self.__windowWidth, self.__numOfSensorsAvailable])  # initialize window
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
                print("(startReceiveData) Recording...")
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
                        line[(line > self.__lineThresholdAfterNorm) | (line < -self.__lineThresholdAfterNorm)] = 0
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
                        dataOutQ = np.zeros([self.__windowWidth, self.__numOfSensorsUsed])
                        for i in range(self.__numOfSensorsUsed):
                            dataOutQ[::, i] = dataOut[::, self.__indexSensorsUsed[i]] * self.__windowFunction

                        # write data to queue:

                        outputQueue.put(dataOutQ)

        if self.__receiveDataP is None and str(opt) != "KILL":
            if self.__windowDataQueue is None:
                self.__windowDataQueue = multiprocessing.Queue()

            if self.__windowDataQueue is None:
                self.__windowDataQueue = multiprocessing.Queue()
            self.__receiveDataP = multiprocessing.Process(target=receiveDataF, args=(self.__windowDataQueue, ))
            self.__receiveDataP.start()
            print("(startReceiveData) Started receiving process...")
        else:
            if str(opt) == "KILL":
                if self.__receiveDataP is not None:
                    self.__receiveDataP.terminate()
                    self.__receiveDataP = None
                    print("(startReceiveData) Terminated receiving process...")
                else:
                    print("(startReceiveData) Receiving process already down...")
            else:
                print("(startReceiveData) ReceiveData-process already started!")

    def startPlotStreamData(self, sensorNr, opt=None):

        """
        Starts a process to plot data of selected sensors (sensorNr). This function takes data directly from the
        receiveDataP via the self.__windowDataQueue (which is initialized if it is not already). Only data of chosen
        sensors can be plotted. The self.__windowDataQueue is of size self.__windowWidth x self.__numOfSensorsUsed.
        :param sensorNr: Single int or array, specifying the sensor indices to be plotted. The int does not refer to
        the original sensor indices, but to the ones, resulting from the sensor selection!
        :param opt: If "KILL" ends process. Default None.
        :return: -
        """

        def plotStreamDataF(plotSensorNr, fl, inputQueue):

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

        if self.__plotStreamDataP is None and opt != "KILL":
            if 0 <= sensorNr < self.__numOfSensorsUsed:

                if self.__windowDataQueue is None:
                    self.__windowDataQueue = multiprocessing.Queue()

                self.__plotStreamDataP = multiprocessing.Process(target=plotStreamDataF, args=(sensorNr,
                                                                                               self.__frameLength,
                                                                                               self.__windowDataQueue))
                self.__plotStreamDataP.start()
                print("(startPlotStreamData) Started process... ")
            else:
                print("\n \n (startPlotStreamData) Invalid sensor number...  \n \n")
        else:
            if opt == "KILL":
                if self.__plotStreamDataP is not None:
                    self.__plotStreamDataP.terminate()
                    self.__plotStreamDataP = None
                    print("(startPlotStreamData) Process terminated!")
                else:
                    print("(startPlotStreamData) Process already down.")
            else:
                print("(startPlotStreamData) Process already started!")

    def startLiveClassification(self, opt=None):

        def liveClassificationF(classifier, featureQ):

            while True:
                features = featureQ.get()
                pred = classifier.predict(features.reshape(1, -1))
                print("(liveClassification) Classification: " + self.__className[int(pred)])

        if self.__liveClassificationP is None and opt != "KILL":
            if self.__trainedClassifier is None:
                print("(startLiveClassification) No classfier trained yet.")
                return False
            else:
                if self.__receiveDataP is None:
                    self.startReceiveData()

                if self.__extractFeaturesP is None:
                    self.initFeatureQueue()
                self.__liveClassificationP = multiprocessing.Process(target=liveClassificationF,
                                                                     args=(self.__trainedClassifier,
                                                                           self.__featureQueue))
                self.__liveClassificationP.start()
                print("(startLiveClassification) Started live-classification process...")
        else:
            if opt == "KILL":
                if self.__liveClassificationP is not None:
                    self.__liveClassificationP.terminate()
                    self.__liveClassificationP = None
                    print("(startLiveClassification) Terminated process.")
                else:
                    print("(startLiveClassification) Process is already down.")
            else:
                print("(startLiveClassification) Process already started!")

    def startOneClass(self, opt=None):

        def liveClassificationF(classifier, featureQ):

            while True:
                features = featureQ.get()
                pred = classifier.predict(features.reshape(1, -1))
                print("(liveOneClassification) Classification: " + str(pred))

        if self.__liveClassificationP is None and opt != "KILL":
            if self.__trainedClassifier is None:
                print("(startOneClass) No classfier trained yet.")
                return False
            else:
                if self.__receiveDataP is None:
                    self.startReceiveData()

                if self.__extractFeaturesP is None:
                    self.initFeatureQueue()
                self.__liveClassificationP = multiprocessing.Process(target=liveClassificationF,
                                                                     args=(self.__trainedClassifier,
                                                                           self.__featureQueue))
                self.__liveClassificationP.start()
                print("(startOneClass) Started live-classification process...")
        else:
            if opt == "KILL":
                if self.__liveClassificationP is not None:
                    self.__liveClassificationP.terminate()
                    self.__liveClassificationP = None
                    print("(startLiveClassification) Terminated process.")
                else:
                    print("(startOneClass) Process is already down.")
            else:
                print("(startOneClass) Process already started!")

    @staticmethod
    def plotMatrixWithValues(matrix, title_=None, precision=3, show=True):

        """
        Plot matrix and its element values
        :param matrix: Matrix (m x n) to be plotted.
        :param title_: Title of the resulting plot, when None no title is given. Default None.
        :param precision: Precision of the element values.
        :param show: Show matrix within function. Default True.
        :return: -
        """

        matrix = np.array(matrix)
        xrange = np.size(matrix[::, 0])
        yrange = np.size(matrix[0, ::])
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.Blues)
        for i in range(yrange):
            for j in range(xrange):
                c = np.round(matrix[j, i], precision)
                ax.text(i, j, str(c), va='center', ha='center')
        if title_ is not None:
            plt.title(title_)
        if show:
            plt.show()


if __name__ == '__main__':

    a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=200, windowShift=200, numFreqs=5, numCoeffs=5,
                          enaStatFeats=True, featNormMethod='stand', kernel='rbf', trainFraction=1, wvltLvl1=False,
                          randomSortTT=False, classSortTT=True)

    a.selectSensorSubset(selectedSensors=[False, False, False], sensorType='bno')
    # a.selectSensorSubset(selectedSensors=[], sensorType='ir')

    # add files:
    # a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=3500, stopTime=3800, label=1,
    #               className="not walking")
    # a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=200, stopTime=500, label=1)
    # a.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=1470, stopTime=1570, label=1)
    # a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=0, stopTime=150, label=1)
    # a.addDataFiles(fileSourceName="markusSchnell.txt", fileSourcePath="../", startTime=4000, stopTime=4300, label=1)
    # a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=7600, stopTime=8600, label=1)
    # a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=0, stopTime=300, label=1)
    # a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=0, stopTime=1000, label=1)
    # a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=7000, stopTime=8000, label=1)
    # a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=5100, stopTime=6000, label=1)

    a.addDataFiles(fileSourceName="igor.txt", fileSourcePath="../", startTime=100, stopTime=2900, label=0,
                   className="walking")
    a.addDataFiles(fileSourceName="igor2.txt", fileSourcePath="../", startTime=600, stopTime=6000, label=0)

    a.addDataFiles(fileSourceName="ankita.txt", fileSourcePath="../", startTime=200, stopTime=1900, label=0)
    a.addDataFiles(fileSourceName="ankita_pos2_lrRl.txt", fileSourcePath="../", startTime=150, stopTime=2500, label=0)

    a.addDataFiles(fileSourceName="chris_asymm.txt", fileSourcePath="../", startTime=200, stopTime=1400, label=0)
    a.addDataFiles(fileSourceName="chris1.txt", fileSourcePath="../", startTime=500, stopTime=5000, label=0)
    a.addDataFiles(fileSourceName="chris_pos2.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=0)

    a.addDataFiles(fileSourceName="chris_c.txt", fileSourcePath="../", startTime=100, stopTime=1700, label=0)

    a.addDataFiles(fileSourceName="markus.txt", fileSourcePath="../", startTime=500, stopTime=4000, label=0)

    a.addDataFiles(fileSourceName="stefan.txt", fileSourcePath="../", startTime=500, stopTime=7000, label=0)

    a.addDataFiles(fileSourceName="ben.txt", fileSourcePath="../", startTime=2000, stopTime=6000, label=0)

    a.setFileSink(fileSinkName="test.txt", fileSinkPath="../")

    a.readDataSet(equalLength=False, checkData=False)

    print("Data read....")

    # a.initFeatNormalization(dumpName="test.pkl")
    a.loadDumpNormParam(dumpName="oneClassNorm")

    # print("Check")

    # clf = svm.OneClassSVM(kernel='rbf')
    # a.trainClassifier(classifier=clf)
    # a.dumpClassifier(dumpName="test")

    a.loadDumpClassifier(dumpName="oneClassClf.pkl")

    # a.startPlotStreamData(sensorNr=0)

    a.startOneClass()


