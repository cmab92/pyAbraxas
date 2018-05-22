########################################################################################################################
## cb, 20.04.18
##
## general info:
## read serial data from txt-file
##
## function inputs:
## fileName: output file name
## dirPath: input file directory
##
## functions output:
## data: data[i,j]:= i-th data point of j-th sensor
########################################################################################################################
import numpy as np
def readSerialFile(fileName, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/"):
    data = np.loadtxt(dirPath + fileName, delimiter=',', comments="%")
    return data