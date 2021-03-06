"""
cb, 27.06.2018

 - receive data via COM port and format data (specifically re-arrange bno055 data)

 Detailed:
 The input data contains data read from arm-analog-ports (infrared sensors and force sensors) and bno055 data (linear
 acceleration, angular velocity and quaternion data). Given is a row of unprocessed values, e. g.:

 ir_0, ir_1, ..., ir_m, force_0, ..., force_n, bno_0, bno_1, bno_2, bno_3, index
 where: 0 <= ir_i, force_j <= 1023 and -2^14 <= bno_k <= 2^14

 The bno data has a lower data rate, to allow higher sampling rates. 'index' indicates the type of data:
 index = 0 := quaternion data
 index = 1 := linear acceleration data
 index = 2 := angular velocity data

 This function receives all data and formats it, s. t. the output data is given as:
 ir_0, ir_1, ..., ir_m, force_0, ..., force_n, q_w, q_x, q_y, q_z, la_x, la_y, la_z, av_y, av_y, av_z
 where: 0 <= ir_i, force_j <= 1 and -1 <= bno_k <= 1

 Incomplete frames are interpolated.

 Optionally a time axis is provided and received data is returned and written to a .txt-file.

Inputs:
numIr           := Number of infrared sensors. Default 10.

numF            := Number of force sensors. Default 2.

filePath        := Path to .txt-file, in which data is stored (if not given, data is stored to the current working
                parent directory). Default "/..".

fileName        := Name of .txt-file. The file is named by yymmddhhmm, if not given. File name is completed with
                yymmddhhmm, if the input string is not ended by .txt. Defaut "".

port            := If no COM-port is specified, /dev/ttyUSB0 ... /dev/ttyUSB7 is checked.

baudRate        := Default 57600.

windowWidth     := Determines the width of the data window in terms of samples. Default 100.

windowShift     := Determines the time shift between data windows in terms of samples. Default 10.

window          := Choose a window function. See also miscFunc.py.

alpha           := Shape parameter of window function (relevant for only some of the window functions). Default 0.1.

dataWindowQueue := Queue object for handing over data windows.

plotNr          := Number of data strings to be plotted. If None, live plotting is disabled. Default None.

 """

import serial
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from abraxasTwo.miscFunc import applyWindow


def receiveData(numIr=10, numF=2, filePath="../", fileName="", port=None, baudRate=57600, windowWidth=100,
                windowShift=10, window="tukey", alpha=0.1, dataWindowQueue=None, plotNr=None):

    #
    # find and connect COM port:
    #

    if port is None:
        for i in range(8):
            port = "/dev/ttyUSB"+str(i)
            try:
                ser = serial.Serial(port=port, baudrate=baudRate)
                break
            except serial.SerialException:
                if i == 7:
                    ser = None
                continue
    else:
        ser = serial.Serial(port=port, baudrate=baudRate)
    try:
        dummy = ser.readline()
    except AttributeError:
        print("\n \n COM-port connected? Baud rate correct? Connected to another process? \n \n ")



    #
    # setup .txt-file:
    #

    dateAndTime = datetime.datetime.now()
    dateAndTime = str(dateAndTime.year) + str(dateAndTime.month) + str(dateAndTime.day) + str(dateAndTime.hour) \
                  + str(dateAndTime.minute)
    if fileName[(len(fileName) - 4):] != ".txt":
        fileName = str(str(fileName) + dateAndTime + ".txt")
    outputFile = open(filePath + fileName, "w")
    with open(file=filePath + fileName, mode="w"):
        writer = csv.writer(outputFile, delimiter=",")
        writer.writerow("%" + "start Time: " + str(datetime.datetime.now().hour) + "h"
                        + str(datetime.datetime.now().minute) + "m" + str(datetime.datetime.now().second) + "s"
                        + str(datetime.datetime.now().microsecond) + "us")

        #
        # format and write data:
        #

        frameLength = numIr + numF + 5                              # 5 for bno data including index (received format)
        windowLength = numIr + numF + 10                            # 10 is for reformatted bno data
        q = [0, 0, 0, 0]                                            # init bno data (quaternion)
        la = [0, 0, 0]                                              # angular velocity
        av = [0, 0, 0]                                              # linear acceleration
        dataWindow = np.zeros([windowWidth, windowLength])          # initialize window
        windowCount = 0                                             # counting samples until next window starts

        oldLine = ser.readline()
        waitCount = 0
        while np.size(oldLine) != frameLength:                      # wait for complete line
            oldLine = ser.readline()
            oldLine = oldLine.decode("utf-8")
            oldLine = oldLine.split(",")[:frameLength]
            if waitCount > 20:                                      # restart connection if stuck
                ser.close()
                ser.open()
        bnoData = 0
        waitCount = 0
        while bnoData == 0:                                         # wait for bno ready / calibration
            oldLine = ser.readline().decode("utf-8").split(",")[:frameLength]
            if np.size(oldLine) == frameLength:
                bnoData = float(oldLine[frameLength - 2]) + float(oldLine[frameLength - 3]) \
                          + float(oldLine[frameLength - 4]) + float(oldLine[frameLength - 5])
            if waitCount > 20:                                      # restart connection if stuck
                ser.close()
                ser.open()
        print("Recording...")
        while True:
            try:
                line = ser.readline().decode("utf-8").split(",")[:frameLength]
                if np.size(line) == frameLength:
                    oldLine = line
                else:                                               # interpolate if message is incomplete
                    line = oldLine
                writer.writerow(line)
                line = np.array(line).astype(np.float)
                if line[frameLength - 1] == 0:
                    for i in range(4):
                        q[i] = line[frameLength - 2 - i]/2**14      # 1 quaternion = 14 LSB
                if line[frameLength - 1] == 1:
                    for i in range(3):
                        la[i] = line[frameLength - 3 - i]/2**4      # normalization constant estimated
                if line[frameLength - 1] == 2:
                    for i in range(3):
                        av[i] = line[frameLength - 3 - i]/2**4      # normalization constant estimated

                line = np.concatenate([line[:(numIr + numF)]/1023, q, la, av])
                line[(line > 10) | (line < -10)] = 0                # threshold estimated (because of la, av)
            except (KeyboardInterrupt, SystemExit, serial.SerialException):
                writer.writerow("%" + "stop Time: " + str(datetime.datetime.now().hour) + "h"
                                + str(datetime.datetime.now().minute) + "m"
                                + str(datetime.datetime.now().second) + "s"
                                + str(datetime.datetime.now().microsecond) + "us")
                break
            windowCount += 1
            dataWindow = np.roll(dataWindow, -1, 0)
            dataWindow[-1, ::] = line
            if windowCount == windowShift - 1:
                windowCount = 0
                dataOut = np.array(dataWindow)
                for i in range(windowLength):
                    dataOut[::, i] = applyWindow(data=dataOut[::, i], window=window, alpha=alpha)

                #
                # live plot data if required:
                #

                if plotNr is not None:
                    if isinstance(plotNr, int):
                        if 0 <= plotNr & plotNr <= frameLength:
                            plt.ion()
                            plt.clf()
                            plt.plot(dataOut[::, plotNr])
                            plt.pause(10**-15)
                    if isinstance(plotNr, (list, tuple, np.ndarray)):
                        plt.ion()
                        plt.clf()
                        for j in range(len(plotNr)):
                            plt.plot(dataOut[::, plotNr[j]])
                        plt.pause(10**-15)

                #
                # write data to queue:
                #

                if dataWindowQueue is not None:
                    try:
                        dataWindowQueue.put(dataOut)
                    except AttributeError:
                        print("\n \n 'dataOutQueue' in function 'receiveData' is not a queue object...\n \n")
                        raise
    return True


if __name__ == '__main__':
    try:
        fileName_ = raw_input("Enter file name: ")                      # start from terminal
        n_ir_entered = 0
        while n_ir_entered == 0:
            n_ir = raw_input("Enter number of ir sensors: ")            # start from terminal
            n_ir_entered = 1
            try:
                n_ir = int(n_ir)
            except ValueError:
                n_ir_entered = 0
        n_force_entered = 0
        while n_force_entered == 0:
            n_force = raw_input("Enter number of force sensors: ")      # start from terminal
            n_force_entered = 1
            try:
                n_force = int(n_force)
            except ValueError:
                n_force_entered = 0
    except:
        fileName_ = "test.txt"
        n_force = 2
        n_ir = 10
    dirPath_ = ""
    receiveData(numIr=10, numF=2, filePath="../", fileName=fileName_, port=None, baudRate=57600, windowWidth=100,
                windowShift=10, window="tukey", alpha=0.1, dataWindowQueue=None, plotNr=None)

