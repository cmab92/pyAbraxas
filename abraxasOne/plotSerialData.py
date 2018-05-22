import serial
import csv
import numpy as np
import matplotlib.pyplot as plt
from readSerialPort import readSerialPort
from readSerialFile import readSerialFile
import threading

##
t1 = threading.Thread()
t2 = threading.Thread()
pointsPerTxt = 100
txtPerWindow = 10
numberOfSensors =17
numberOfDataPoints = int(pointsPerTxt*txtPerWindow)
data = np.zeros((numberOfDataPoints,numberOfSensors))
plt.ion()
while(1):
    readSerialPort("livePlot.txt", numberOfSensors, dataPoints=pointsPerTxt)
    newData = readSerialFile("livePlot.txt")
    data = np.roll(data, int(pointsPerTxt*(txtPerWindow-1)),0)
    data[(numberOfDataPoints-pointsPerTxt):,::] = newData
    plt.figure(1)
    plt.clf()
    plt.ylim([0, 500])
    plt.plot(data[::,0],'r')
    plt.plot(data[::,1],'g')
    plt.plot(data[::,2],'b')
    plt.plot(data[::,3],'y')
    plt.pause(0.00000001)
plt.show()