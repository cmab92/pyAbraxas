import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from abraxasOne.helperFunctions import writeMatrixToCsvFile

from fastdtw import fastdtw

'params'
fs = 44*10**3
Ts = 1/fs
fsig = fs/10/4


amp = 1
l1 = 0.001
l2 = 0.002

'sig gen'

x1 = np.linspace(0, l1, l1/Ts)
x2 = np.linspace(0, l2, l2/Ts)
y1 = amp*np.sin(2*np.pi*fsig*x1)
y2 = amp*np.sin(2*np.pi*fsig*x2)
# y1 = 10**10*y1
# y1[y1<0] = -1
# y1[y1>0] = 1
print("Power density = " + str(np.sum(np.square(y1/(np.sum(np.square(y1)))))))
distance, path = fastdtw(y1/(np.sum(np.square(y1))), y2/(np.sum(np.square(y2))), dist=euclidean)
distance, path = fastdtw(y1, y2, dist=euclidean)
path = np.array(path)
print(distance)

stretched = []
for i in range(len(path[::, 0])):
    stretched.append(y1[path[i, 0]])

plt.figure()
stretched = np.array(stretched)
plt.plot(x2, stretched)
plt.plot(x2, y2)

'landscape'

M = []
for i in range(len(y2)):
    temp = []
    for j in range(len(y1)):
        temp.append(np.sqrt(np.square(y2[i] - y1[j])))
    M.append(np.array(temp).T)
    # M.append(euclidean(y2[i]*np.ones(len(y1)),y1))
writeMatrixToCsvFile(data=np.array(M).T, fileName="dtwSin.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/")

# check gangmusteranalyse.tex to unserstand (sin should be plotted to axis)

writeMatrixToCsvFile(data=path, fileName="dtwSinPath.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
axis1 = np.linspace(0, len(y1)-1, len(y1))
axis2 = np.linspace(0, len(y2)-1, len(y2))

y1 = -3*y1 - 5*np.ones(len(axis1))
y2 = 2*y2 - 5*np.ones(len(axis2))

writeMatrixToCsvFile(data=np.array([axis1, y1]).T, fileName="dtwSinX.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
writeMatrixToCsvFile(data=np.array([y2, axis2]).T, fileName="dtwSinY.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)

# axis2 = axis2/5 + 50
# y2 = (y2 - np.mean(y2))/np.max(y2)
# stretched = (stretched - np.mean(stretched))/np.max(stretched)
# y2 = 15*y2 + 15
# stretched = 15*stretched + 15

# writeMatrixToCsvFile(data=np.array([axis2, y2]).T, fileName="dtwSinOrig.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)
# writeMatrixToCsvFile(data=np.array([axis2, stretched]).T, fileName="dtwSinStretched.dat", path="/home/bonenberger/Dokumente/Praesentationen/Gangmusteranalyse/", matrix=False)

'plotting'
plt.figure()
plt.plot(path[::, 0], path[::, 1], 'r', label="Cost = " + str(np.round(distance, 2)))
plt.legend()
plt.imshow(M)
plt.figure()
plt.plot(np.linspace(0, 1, len(y1)), y1)
plt.plot(np.linspace(0, 1, len(y2)), y2)
plt.show()
