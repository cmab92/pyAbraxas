"""
cb, 28.06.2018

 - apply window function to a vector

Inputs:
data            := Input data.

window          := Choose a window function of the following (https://en.wikipedia.org/wiki/Window_function):
                tukey -> tukey window (flattened cosine)
                rect -> rectangular window
                bart -> bartlett window
                black -> blackman window
                ham -> hamming window
                hann -> hanning window (raised-cosine window)
                kaiser -> kaiser window
                gauss -> gaussian window
                Default "tukey".

alpha           := Shape parameter of window function (not relevant for all). Default 0.1.

samplingPeriod  := Giving the sampling period (in seconds) throws a plot of the window response in frequency domain.
                Default None.

Output:


"""
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def applyWindow(data, window="tukey", alpha=0.1, samplingPeriod=None):

    data = np.array(data)
    windowWidth = np.size(data)

    if window == 'tukey':
        windowFunction = scipy.signal.tukey(windowWidth, alpha)
    elif window == 'rect':
        windowFunction = np.ones(windowWidth)
    elif window == 'bart':
        windowFunction = np.bartlett(windowWidth)
    elif window == 'black':
        windowFunction = np.blackman(windowWidth)
    elif window == 'ham':
        windowFunction = np.hamming(windowWidth)
    elif window == 'hann':
        windowFunction = np.hanning(windowWidth)
    elif window == 'kaiser':
        windowFunction = np.kaiser(windowWidth, alpha)
    elif window == 'gauss':
        windowFunction = scipy.signal.gaussian(windowWidth, alpha)
    else:
        windowFunction = np.ones(windowWidth)

    if samplingPeriod is not None:
        if isinstance(samplingPeriod, float):
            timeAxis = np.linspace(0, windowWidth*samplingPeriod, windowWidth)
            plt.plot(timeAxis, windowFunction)
            plt.title('Time Function of ' + str(window) + ' window')
            plt.xlabel('t in s')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.figure()
            freqAxis = np.linspace(0, 2/samplingPeriod, int(windowWidth))
            windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(
                np.concatenate((np.zeros(int(windowWidth/2)), windowFunction, np.zeros(int(windowWidth/2)))))))
            windowFreqResponse = 20*np.log10(windowFreqResponse/np.max(windowFreqResponse))
            plt.plot(freqAxis, windowFreqResponse[int(windowWidth):])
            plt.xlim(0,)
            plt.ylim(-120,)
            plt.title("Frequency Response of chosen Window (" + str(window) + ", Conv. Th.)")
            plt.xlabel('f in Hz')
            plt.ylabel('dB')
            plt.grid()
            plt.show()

    return data*windowFunction

"""
cb, 10.07.2018

 - plot matrix and its element values
 
Input:
matrix      := Matrix (m x n) to be plotted.

title_      := Title of the resulting plot, when None no title is given. Default None.

precision   := Precision of the element values.

show        := Show matrix within function. Default True.
"""


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
