import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from abraxasOne.loadAndClean import loadAndClean
irData, forceData, quatData, linAccData, angVecData = loadAndClean("igor.txt", 10, 2, tSample=0.0165, dirPath="")
start = 100
stop = 2000
x11 = irData[0][start:stop,1]
x21 = irData[3][start:stop,1]
x31 = irData[5][start:stop,1]
irData, forceData, quatData, linAccData, angVecData = loadAndClean("ankita.txt", 10, 2, tSample=0.0165, dirPath="")
start = 100
stop = 1800
x12 = irData[0][start:stop,1]
x22 = irData[3][start:stop,1]
x32 = irData[5][start:stop,1]
irData, forceData, quatData, linAccData, angVecData = loadAndClean("chris_pos2.txt", 10, 2, tSample=0.0165, dirPath="")
start = 200
stop = 1600
x13 = irData[0][start:stop,1]
x23 = irData[3][start:stop,1]
x33 = irData[5][start:stop,1]
irData, forceData, quatData, linAccData, angVecData = loadAndClean("igor2.txt", 10, 2, tSample=0.0165, dirPath="")
start = 2500
stop = 4250
x14 = irData[0][start:stop,1]
x24 = irData[3][start:stop,1]
x34 = irData[5][start:stop,1]

plt.figure()
plt.plot(x11, label='x11')
plt.plot(x12, label='x12')
plt.plot(x13, label='x13')
plt.plot(x14, label='x14')
plt.legend()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x11, x21, x31, c='r', alpha=0.1, label="1")
ax.scatter(x12, x22, x32, c='g', alpha=0.1, label="2")
ax.scatter(x13, x23, x33, c='b', alpha=0.1, label="3")
ax.scatter(x14, x24, x34, c='y', alpha=0.1, label="4")
plt.legend()
plt.show()
