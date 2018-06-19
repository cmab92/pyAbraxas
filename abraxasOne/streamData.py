########################################################################################################################
## cb, 20.04.18
##
## general info:
## read data from serial port
##
## function inputs:
## fileName: output file name... if .txt no time and date... else time and date attached
## numOfSensors: number of analog (!) Sensors
## dirpath: output file directory
## baudRate: ...
## dataPoints: number of data points to get
## port: serial port
##
########################################################################################################################
import serial
import csv
import numpy as np
import datetime
import matplotlib.pyplot as plt
from loadAndClean import loadAndClean
##
def streamData(numOfSensors, windowsize, windowshift, baudRate = 57600, port = '/dev/ttyUSB0'):
    ser = serial.Serial(port, baudRate)
    dummy = ser.readline() # dummy read ... read first (possibly incomplete) line
    oldLine = ser.readline() # get line for interpolation
    oldLine = oldLine.decode("utf-8")
    numOfSensors = numOfSensors+5
    while (np.size(oldLine)!=numOfSensors): # wait until received complete message
        oldLine = ser.readline()
        oldLine = oldLine.decode("utf-8")
        oldLine = oldLine.split(",")[:numOfSensors]
    window = np.zeros([windowsize, numOfSensors])
    count = 0
    while(1):
        count +=1
        line = ser.readline()
        line = line.decode("utf-8")
        line = line.split(",")[:numOfSensors]
        if (np.size(line)==numOfSensors):
            oldLine = line
        else: # interpolate if message is incomplete
            line = oldLine
        window = np.roll(window, -1, 0)
        window[-1, ::] = line
        if count>=windowshift:
            count = 0
            plt.figure()
            plt.plot(window[::, 0])
            plt.pause(10**-15)

plt.ion()
streamData(10, 1000, 100)