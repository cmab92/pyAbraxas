########################################################################################################################
## cb, 20.04.18
##
## general info:
## read data from serial port
##
## function inputs:
## fileName: output file name
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
##
def readSerialPort(fileName, numOfSensors, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/", baudRate = 57600, dataPoints = 'inf', port = '/dev/ttyUSB0'):
    dateAndTime = datetime.datetime.now()
    dateAndTime = str(dateAndTime.year) + str(dateAndTime.month) + str(dateAndTime.day) + str(dateAndTime.hour) + str(dateAndTime.minute)
    if (fileName[(len(fileName)-4):]==".txt"):
        fileName = fileName
    else:
        fileName = str(str(fileName) + "_" + dateAndTime + ".txt")
    outputFile = open(dirPath + fileName, "w")
    ser = serial.Serial(port, baudRate)
    dummy = ser.readline() # dummy read ... read first (possibly incomplete) line
    oldLine = ser.readline() # get line for interpolation
    oldLine = oldLine.decode("utf-8")
    while (np.size(oldLine)!=numOfSensors): # wait until received complete message
        oldLine = ser.readline()
        oldLine = oldLine.decode("utf-8")
        oldLine = oldLine.split(",")[:numOfSensors]
    with open(dirPath + fileName, "w"):
        writer = csv.writer(outputFile, delimiter=" ")
        writer.writerow("%" + "start Time: " + str(datetime.datetime.now().hour) + "h" + str( #...
            datetime.datetime.now().minute) + "m" + str(datetime.datetime.now().second) + "s" + str(datetime.datetime.now().microsecond) + "us")
        writer = csv.writer(outputFile, delimiter=",")
        if (isinstance(dataPoints, int)==0):
            while True:
                try:
                    line = ser.readline()
                    line = line.decode("utf-8")
                    line = line.split(",")[:numOfSensors]
                    if (np.size(line)==numOfSensors):
                        oldLine = line
                    else: # interpolate if message is incomplete
                        line = oldLine
                    writer.writerow(line)
                except (KeyboardInterrupt, SystemExit):
                    writer.writerow("%" + "stop Time: " + str(datetime.datetime.now().hour) + "h" + str( #...
                    datetime.datetime.now().minute) + "m" + str(datetime.datetime.now().second) + "s")
                    raise
        else:
            count = 0
            while (count < dataPoints):
                line = ser.readline()
                line = line.decode("utf-8")
                line = line.split(",")[:numOfSensors]
                if (np.size(line)==numOfSensors):
                    oldLine = line
                else: # interpolate if message is incomplete
                    line = oldLine
                writer.writerow(line)
                count += 1
            writer.writerow("%" + "stop Time: " + str(datetime.datetime.now().hour) + "h" + str( #...
                datetime.datetime.now().minute) + "m" + str(datetime.datetime.now().second) + "s" + str(datetime.datetime.now().microsecond) + "us")
    return True
## call from terminal / ide:
if __name__ == '__main__':
    try:
        fileName_ = raw_input("Enter file name: ") # start from terminal
    except:
        fileName_ = "test"
    dirPath_ = ""
    print("Recording...")
    readSerialPort(fileName=fileName_, numOfSensors=17, dirPath=dirPath_, baudRate=57600, dataPoints="inf")