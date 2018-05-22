from qtPlotSerialData import qtPlotSerialData
from readSerialFile import readSerialFile
from sliceAndWindow import sliceAndWindow
from loadAndClean import loadAndClean
import matplotlib.pyplot as plt
import numpy as np
irData, forceData, quatData, linAccData, angVecData = loadAndClean("_20185161226.txt", 10, 4, tSample = 0.0165, dirPath = "/home/bonenberger/Dokumente/Rabe/Daten/dataRABE/")
plt.plot(angVecData[0][::,1]*10)
plt.title("av")
plt.figure()
plt.plot(linAccData[0][::,1]*10)
plt.title("la")
plt.show()