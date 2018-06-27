import multiprocessing
import serial
from abraxasOne.extractFeatures import extractFeatures
from abraxasOne.helperFunctions import scaleData
from trainSVC import trainSVC
from six.moves import cPickle
import scipy.signal
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from six.moves import cPickle
from  abraxasOne.splitDataTrainTest import splitDataTrainTest
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues

def getFeaturesF(plotDataQ, featureDataQ, totalNumOfSensors, usedSensors, windowWidth, windowShift, numDomCoeffs,
                 numDomFreqs, window, alpha, useBno):
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

    normData = np.loadtxt("normData.txt", delimiter=",")
    mean = np.array(normData[0, ::])
    std_dev = np.array(normData[1, ::])
    quat = ['0', '0', '0', '0']
    linAcc = ['0', '0', '0']
    angVec = ['0', '0', '0']
    dummy = ser.readline() # get line for interpolation
    oldLine = ser.readline() # get line for interpolation
    oldLine = oldLine.decode("utf-8")
    dataWindow = np.zeros([windowWidth, np.size(usedSensors)])
    if useBno:
        dataWindow = np.zeros([windowWidth, np.size(usedSensors)+10])
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
            if useBno:
                if line[16]=='0':
                    for j in range(4):
                        quat[j] = str(float(line[12+j])/2**14*500)
                if line[16]=='1':
                    for j in range(3):
                        linAcc[j] = str(float(line[12+j])/2**8*500)
                if line[16]=='2':
                    for j in range(3):
                        angVec[j] = str(float(line[12+j])/2**8*500)
                for j in range(4):
                    newLine.append(quat[j])
                for j in range(3):
                    newLine.append(linAcc[j])
                for j in range(3):
                    newLine.append(angVec[j])
            dataWindow = np.roll(dataWindow, -1, 0)
            dataWindow[-1, ::] = np.array(newLine)
            print(newLine)
        dataOut = np.array(dataWindow)
        for i in range(np.size(newLine)):
            dataOut[::, i] = dataOut[::, i]*windowFunction
        f = extractFeatures(dataOut, numDomCoeffs=numDomCoeffs, numDomFreqs=numDomFreqs, wavelet='haar', statFeat=True)
        for j in range(np.size(f)):
            f[j] = (f[j] - mean[j])/(std_dev[j])
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
    if 0:
        files = ["igor.txt", "chris_c.txt", "ankita_pos2_lrRl.txt", "igor2.txt", "chris1.txt", "chris_pos2.txt", "ben.txt",
                 "markus.txt", "markusSchnell.txt"]
        start = np.array([600, 100, 100, 3500, 500, 100, 2000, 500, 100])
        stop = np.array([3400, 1600, 3000, 6000, 4500, 1400, 5500, 3500, 1400])
        numberOfClasses = 6

        fileLabels = np.array([0, 1, 2, 0, 3, 3, 4, 5, 5])
        fileLabelsSym = ['igor, 0', 'crooked, 1', 'ankita, 2', 'chris, 3', 'ben, 4', 'markus, 5']

        usedSensors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        trF, teF, trL, teL, mean, std_dev = splitDataTrainTest(files, start, stop, fileLabels, windowWidth=100,
                                                               windowShift=10,
                                                               numDomCoeffs=10, numDomFreqs=0, trainFrac=0.99,
                                                               statFeat=True,
                                                               shuffleData=False, checkData=False)
    if 0:
        classifier = svm.SVC(kernel='rbf')  # , C=100.0236, gamma=10.00227)
        classifier.fit(trF, trL)

    with open('dumpedSVC.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)

    plotDataQ = multiprocessing.Queue()
    featureDataQ = multiprocessing.Queue()

    totalNumOfSensors = 10+2+5
    usedSensors = np.array([0, 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11])
    windowWidth = 100
    windowShift = 10
    numDomCoeffs = 10
    numDomFreqs = 0
    analogPort = 16
    window = 'tukey'
    alpha = 0.1
    useBno = True

    featureProcess = multiprocessing.Process(target=getFeaturesF, args=(plotDataQ, featureDataQ, totalNumOfSensors,
                                                                        usedSensors, windowWidth, windowShift,
                                                                        numDomCoeffs, numDomFreqs, window, alpha,
                                                                        useBno, ))
    classProcess = multiprocessing.Process(target=classDataF, args=(plotDataQ, featureDataQ, classifier, ))

    plotProcess = multiprocessing.Process(target=plotDataF, args=(plotDataQ, analogPort, ))

    featureProcess.start()

    #plotProcess.start()

    classProcess.start()
