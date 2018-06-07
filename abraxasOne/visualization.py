import matplotlib.pyplot as plt
import numpy as np

from abraxasOne.loadAndClean import loadAndClean
from abraxasOne import helperFunctions as hf

data, dummy, dummy, dummy ,dummy =loadAndClean(fileName="../ankita.txt", numberOfIrSensors=10, numberOfForceSensors=2, tSample=0.0165, dirPath="")

# ommit time
for i in range(len(data)):
    data[i] = data[i][130:2000,1]

data = hf.scaleData(data=data, scaleAmp=1, scaleMean=1, scaleVar=0)

plt.plot(data[0][::])
plt.show()
