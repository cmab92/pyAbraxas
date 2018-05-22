import matplotlib.pyplot as plt
import numpy as np
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