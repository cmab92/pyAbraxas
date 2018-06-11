import matplotlib.pyplot as plt
import numpy as np
import random

def scaleData(data, scaleAmp=True, scaleMean=True, scaleVar=True):
    for i in range(len(data)):
        if scaleAmp:
            data[i] = (data[i]-np.min(data[i]))/(np.max(data[i]-np.min(data[i])))
        if scaleMean:
            data[i] = data[i] - np.mean(data[i])
        if scaleVar:
            data[i] = data[i] / np.sqrt(np.var(data[i]))
    return data

def scalogram(data = 0 , title = 'default'):
    bottom = 0

    vmin = min(map(lambda x: min(abs(x)), data))
    vmax = max(map(lambda x: max(abs(x)), data))

    plt.gca().set_autoscale_on(False)
    plt.title(title)


    for row in range(0, len(data)):
        scale = 2.0 ** (row - len(data))

        plt.imshow(
            np.array([(data[row])]),
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1, bottom, bottom + scale])
#            extent=[time[0], time[-1], bottom, bottom + scale])

        bottom += scale

def shuffleData(x, y):
    if len(x)==len(y):
        x_out = np.zeros(np.shape(x))
        y_out = np.zeros(np.shape(y))
        index = np.linspace(0, len(x) - 1, len(x))
        random.shuffle(index)

        for i in range(len(index)):
            x_out[i] = x[int(index[i])]
            y_out[i] = y[int(index[i])]
    else:
        print("Data not shuffled (x!=y).")
    return x_out, y_out