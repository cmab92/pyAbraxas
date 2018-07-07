import multiprocessing
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from abraxasOne.extractFeatures import extractFeatures
from abraxasOne.helperFunctions import scaleData
from trainSVC import trainSVC
from six.moves import cPickle
import scipy.signal

def getFeaturesF(plotDataQ, featureDataQ, totalNumOfSensors, usedSensors, windowWidth, windowShift, numDomCoeffs, numDomFreqs, window, alpha):
    try:
        ser = serial.Serial(port="/dev/ttyUSB0", baudrate=57600)
    except serial.SerialException:
        try:
            ser = serial.Serial(port="/dev/ttyUSB1", baudrate=57600)
        except serial.SerialException:
            try:
                ser = serial.Serial(port="/dev/ttyUSB2", baudrate=57600)
            except serial.SerialException:
                try:
                    ser = serial.Serial(port="/dev/ttyUSB3", baudrate=57600)
                except serial.SerialException:
                    print("USB connected???")
                    ser = serial.Serial(port="/dev/ttyUSB0", baudrate=57600)
    dummy = ser.readline() # throw first line
    oldLine = ser.readline() # get line for interpolation
    oldLine = oldLine.decode("utf-8")
    dataWindow = np.zeros([windowWidth, np.size(usedSensors)])
    data = np.loadtxt("" + "chris1.txt", delimiter=',', comments="%") # data: data[i,j]:= i-th data point of j-th sensor
    count = 0
    while(1):
        if (window == 'tukey'):
            windowFunction = scipy.signal.tukey(windowWidth, alpha)
        elif (window == 'rect'):
            windowFunction = np.ones(windowWidth)
        elif (window == 'bart'):
            windowFunction = np.bartlett(windowWidth)
        elif (window == 'black'):
            windowFunction = np.blackman(windowWidth)
        elif (window == 'ham'):
            windowFunction = np.hamming(windowWidth)
        elif (window == 'hann'):
            windowFunction = np.hanning(windowWidth)
        elif (window == 'kaiser'):
            windowFunction = np.kaiser(windowWidth, alpha)
        elif (window == 'gauss'):
            windowFunction = scipy.signal.gaussian(windowWidth, alpha)
        else:
            windowFunction = np.ones(windowWidth)

        for i in range(windowShift):
            line = ser.readline()
            line = line.decode("utf-8")
            line = line.split(",")[:totalNumOfSensors]
            if (np.size(line)==totalNumOfSensors):
                oldLine = line
            else: # interpolate if message is incomplete
                line = oldLine
            newLine = []
            for j in range(np.size(usedSensors)):
                newLine.append(line[usedSensors[j]])
            dataWindow = np.roll(dataWindow, -1, 0)
            dataWindow[-1, ::] = np.array(newLine)
        #testWithTxt = 0
        #if testWithTxt==1:
        #    for i in range(windowShift):
        #        count += 1
        #        line = data[count,::]
        #        newLine = []
        #        for j in range(np.size(usedSensors)):
        #            newLine.append(line[usedSensors[j]])
        #        dataWindow = np.roll(dataWindow, -1, 0)
        #        dataWindow[-1, ::] = np.array(newLine)
        dataOut = np.array(dataWindow)
        for i in range(np.size(usedSensors)):
            dataOut[::, i] = dataOut[::, i]*windowFunction
        #dataOut = scaleData(dataOut, scaleVar=True, scaleAmp=True, scaleMean=False)
        f = extractFeatures(dataOut, numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, wavelet='haar')
        featureDataQ.put(f)
        plotDataQ.put(dataOut)

def plotDataF(plotDataQ, analogPort):
    plt.figure()
    plt.ion()
    while(1):
        plt.clf()
        dataWindow = plotDataQ.get()
        plt.plot(dataWindow[::, analogPort])
        plt.pause(10**-12)

def classDataF(plotDataQ, featureDataQ, classifier):
    fileLabelsSym = ['igor, 0', 'ankita, 1', 'chris, 2', 'crooked, 3', 'ben, 4', 'markus, 5',
                     'schnell (markus), 6']
    while(1):
        features = featureDataQ.get()
        pred = classifier.predict(features.reshape(1, -1))
        print(fileLabelsSym[int(pred)])
        #plt.clf()
        #dataWindow = plotDataQ.get()
        #plt.plot(dataWindow[::, analogPort])
        #plt.title(fileLabelsSym[int(pred)])
        #plt.pause(10**-12)

if __name__ == '__main__':

    #classifier = trainSVC()

    with open('my_dumped_classifier.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)

    plotDataQ = multiprocessing.Queue()
    featureDataQ = multiprocessing.Queue()

    totalNumOfSensors = 10+2+5
    usedSensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    windowWidth = 150
    windowShift = 10
    numDomCoeffs = 20
    numDomFreqs = 20
    analogPort = 10
    window = 'tukey'
    alpha = 0.1

    featureProcess = multiprocessing.Process(target=getFeaturesF, args=(plotDataQ, featureDataQ, totalNumOfSensors, usedSensors, windowWidth, windowShift, numDomCoeffs, numDomFreqs, window, alpha, ))
    plotProcess = multiprocessing.Process(target=plotDataF, args=(plotDataQ, analogPort, ))
    classProcess = multiprocessing.Process(target=classDataF, args=(plotDataQ, featureDataQ, classifier, ))

    featureProcess.start()
    plotProcess.start()
    #classProcess.start()
