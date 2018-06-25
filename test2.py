import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0, np.pi, 1001)
y1 = np.sin(2*x) + np.random.rand(np.size(x))
y2 = np.sin(2*x-np.pi) + np.random.rand(np.size(x))
plt.plot(x, y1)
plt.plot(x, y2)
plt.figure()
plt.plot(np.correlate(y1, y2, mode='same'))
plt.show()