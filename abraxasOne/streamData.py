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
##
windowWidth = 100
windowShift = 10
numOfSensors = 10
ser = serial.Serial(port="/dev/ttyUSB0", baudrate=57600)
dummy = ser.readline() # throw first line
window = np.zeros([100, numOfSensors])
count = 0
while(1):
    count += 1
    line = ser.readline()
    line = line.decode("utf-8")
    line = line.split(",")[:numOfSensors]
    window = np.roll(window, -1, 0)
    window[-1, ::] = line
    if count>=windowShift:
        print(window)