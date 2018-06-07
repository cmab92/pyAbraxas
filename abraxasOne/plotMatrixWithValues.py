########################################################################################################################
## cb, 05.064.18
##
## general info:
## see fctname
##
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

def plotMatrixWithValues(matrix, title_=None, precision=3, show=True):
    matrix = np.array(matrix)
    xrange = np.size(matrix[::,0])
    yrange = np.size(matrix[0,::])
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)
    for i in range(yrange):
        for j in range(xrange):
            c = np.round(matrix[j,i], precision)
            ax.text(i, j, str(c), va='center', ha='center')
    if title_!=None:
        plt.title(title_)
    if show:
        plt.show()