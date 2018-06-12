########################################################################################################################
## cb, 05.04.18
##
## general info:
## select data slices (training / test / ...) split data to windows (50% overlap)
##
## function inputs:
## data: data[i][j,k] -> i-th sensor, j-th data-point (k=0: time axis, k=1 sensor-value, time in seconds)
## startT: start point of slice (in time [s])
## stopT: end point of slice (" " ")
## windowWidth: width of data windows in seconds (with respect to sampleT)
## sampleT: sampling period time in seconds
## enaCheck: enables plotting of chosen slices
## window: choice of window function (https://en.wikipedia.org/wiki/Window_function)
##      - tukey -> tukey window (flattened cosine)
##        rect -> rectangular window
##        bart -> bartlett window
##        black -> blackman window
##        ham -> hamming window
##        hann -> hanning window (raised-cosine window)
##        kaiser -> kaiser window
##        gauss -> gaussian window
## alpha: shape parameter for tukey, gaussian and kaiser window
## enaCWF: show window function and fft
##
## functions output:
## data: data[i][j,k] -> i-th window, j-th data point of k-th sensor
## numOfWindows: number of returned data windows (for convenience)
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage
##
def sliceAndWindow(data, startT, stopT, windowWidth, sampleT=0.012, enaCheck=1, window='tukey', alpha=0.1, enaCWF=0):
    ## cut slice from input-data:
    numOfSensors = len(data[::])
    dataFrame = []
    for i in range(numOfSensors):
        selectedTime = []
        selectedData = []
        dataFrameTemp = []
        for j in range(np.size(data[i][::, 0])):
            if ((data[i][j, 0] > startT) & (data[i][j, 0] < stopT)):
                selectedTime.append(data[i][j, 0])
                selectedData.append(data[i][j, 1])
        dataFrameTemp.append(selectedTime)
        dataFrameTemp.append(selectedData)
        dataFrame.append(np.transpose(dataFrameTemp))
    data = dataFrame
    if enaCheck:
        plt.figure()
        for i in range(numOfSensors):
            plt.plot(data[i][::, 0], data[i][::, 1], label = i)
        plt.legend()
        plt.title('Check selected data!')
        plt.show()
    ## cut slices for windows
    windowNumberOfPoints = int(windowWidth / (2*sampleT))*2
    numOfWindows = int(np.size(data[0][::, 0]) / windowNumberOfPoints)
    totalNumberOfPoints = int(windowNumberOfPoints * numOfWindows)  # total number of points in dataset
    ## choice of window function:
    if (window=='tukey'):
        windowFunction = scipy.signal.tukey(windowNumberOfPoints, alpha)
    elif (window=='rect'):
        windowFunction = np.ones(windowNumberOfPoints)
    elif (window=='bart'):
        windowFunction = np.bartlett(windowNumberOfPoints)
    elif (window=='black'):
        windowFunction = np.blackman(windowNumberOfPoints)
    elif (window=='ham'):
        windowFunction = np.hamming(windowNumberOfPoints)
    elif (window=='hann'):
        windowFunction = np.hanning(windowNumberOfPoints)
    elif (window=='kaiser'):
        windowFunction = np.kaiser(windowNumberOfPoints, alpha)
    elif (window=='gauss'):
        windowFunction = scipy.signal.gaussian(windowNumberOfPoints, alpha)
    else:
        print("Rectangular Window!")
        windowFunction = np.ones(windowNumberOfPoints)
    if enaCWF==1:
        timeAxis = np.linspace(0,windowWidth, windowNumberOfPoints)
        plt.plot(timeAxis, windowFunction)
        plt.title('Time Function of chosen Window')
        plt.xlabel('t in s')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.figure()
        freqAxis = np.linspace(0,2/sampleT,int(windowNumberOfPoints))
        windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(np.concatenate( (np.zeros(int(windowNumberOfPoints/2)),windowFunction,np.zeros(int(windowNumberOfPoints/2)) )))))
        windowFreqResponse = 20*np.log10(windowFreqResponse/np.max(windowFreqResponse))
        plt.plot(freqAxis,windowFreqResponse[int(windowNumberOfPoints):])
        plt.xlim(0,)
        plt.ylim(-120,)
        plt.title('Frequency Response of chosen Window (Convolution Theorem)')
        plt.xlabel('f in Hz')
        plt.ylabel('dB')
        plt.grid()
        plt.show()
    ## apply window:
    windowedData1 = []
    windowedData2 = []
    # generate concatenated windows (with length of data)
    wFct = windowFunction
    wFct1 = wFct
    wFct2 = wFct[int(windowNumberOfPoints / 2):]
    for i in range(numOfWindows - 1):
        wFct1 = np.concatenate((wFct1, wFct))
        wFct2 = np.concatenate((wFct2, wFct))
    wFct2 = np.concatenate((wFct2, wFct[:int(windowNumberOfPoints / 2)]))
    for i in range(numOfSensors):
        dataTemp = data[i][::, 1]
        dataTemp2 = dataTemp[:totalNumberOfPoints]
        windowedData1.append(dataTemp2 * wFct1)
        windowedData2.append(dataTemp2 * wFct2)
    windowedData1 = np.transpose(windowedData1)
    windowedData2 = np.transpose(windowedData2)
    windows = []
    for i in range(numOfWindows):
        windows.append(windowedData1[int(i * windowNumberOfPoints):(int((i + 1) * windowNumberOfPoints))])
        windows.append(windowedData2[int(i * windowNumberOfPoints + windowNumberOfPoints / 2):(int((i + 1) * windowNumberOfPoints + windowNumberOfPoints / 2))])
    data = windows[:int(numOfWindows)]  # omit last window
    return data, numOfWindows

## version 2
########################################################################################################################
## cb, 10.04.18
##
## general info:
## select data slices (training / test / ...) split data to windows
## new data window each x time steps ...
##
## function inputs:
## data: data[i][j,k] -> i-th sensor, j-th data-point in time... for k=0: ros-time axis, k=1 distance-value... time in seconds!!!
## startT: start point of slice (in time [s])
## stopT: end point of slice (" " ")
## windowWidth: width of data windows in seconds (with respect to sampleT)
## windowShift: after how many time steps a new window starts?
## sampleT: sampling period time in seconds
## enaCheck: enables plotting of chosen slices
## window: choice of window function (https://en.wikipedia.org/wiki/Window_function)
##      - tukey -> tukey window (flattened cosine)
##        rect -> rectangular window
##        bart -> bartlett window
##        black -> blackman window
##        ham -> hamming window
##        hann -> hanning window (raised-cosine window)
##        kaiser -> kaiser window
##        gauss -> gaussian window
## alpha: shape parameter for tukey, gaussian and kaiser window
## enaCWF: show window function and fft
##
## functions output:
## data: data[i][j,k] -> i-th window, j-th data point of k-th sensor
## numOfWindows: number of returned data windows (for convenience)
##
########################################################################################################################
##
def sliceAndWindowV2(data, startT, stopT, windowWidth, windowShift, sampleT=0.012, enaCheck=1, window='tukey', alpha=0.1, enaCWF=0):
    ## cut slice from input-data:
    numOfSensors = len(data[::])
    dataFrame = []
    for i in range(numOfSensors):
        selectedTime = []
        selectedData = []
        dataFrameTemp = []
        for j in range(np.size(data[i][::, 0])):
            if ((data[i][j, 0] > startT) & (data[i][j, 0] < stopT)):
                selectedTime.append(data[i][j, 0])
                selectedData.append(data[i][j, 1])
        dataFrameTemp.append(selectedTime)
        dataFrameTemp.append(selectedData)
        dataFrame.append(np.transpose(dataFrameTemp))
    data = dataFrame
    if enaCheck:
        plt.figure()
        for i in range(numOfSensors):
            plt.plot(data[i][::, 0], data[i][::, 1], label = i)
        plt.legend()
        plt.title('Check selected data!')
        plt.show()
    ## cut slices for windows
    windowNumberOfPoints = int(windowWidth / (2*sampleT))*2
    ## choice of window function:
    if (window=='tukey'):
        windowFunction = scipy.signal.tukey(windowNumberOfPoints, alpha)
    elif (window=='rect'):
        windowFunction = np.ones(windowNumberOfPoints)
    elif (window=='bart'):
        windowFunction = np.bartlett(windowNumberOfPoints)
    elif (window=='black'):
        windowFunction = np.blackman(windowNumberOfPoints)
    elif (window=='ham'):
        windowFunction = np.hamming(windowNumberOfPoints)
    elif (window=='hann'):
        windowFunction = np.hanning(windowNumberOfPoints)
    elif (window=='kaiser'):
        windowFunction = np.kaiser(windowNumberOfPoints, alpha)
    elif (window=='gauss'):
        windowFunction = scipy.signal.gaussian(windowNumberOfPoints, alpha)
    if enaCWF==1:
        timeAxis = np.linspace(0,windowWidth, windowNumberOfPoints)
        plt.plot(timeAxis, windowFunction)
        plt.title('Time Function of chosen Window')
        plt.xlabel('t in s')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.figure()
        freqAxis = np.linspace(0,2/sampleT,int(windowNumberOfPoints))
        windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(np.concatenate( (np.zeros(int(windowNumberOfPoints/2)),windowFunction,np.zeros(int(windowNumberOfPoints/2)) )))))
        windowFreqResponse = 20*np.log10(windowFreqResponse/np.max(windowFreqResponse))
        plt.plot(freqAxis,windowFreqResponse[int(windowNumberOfPoints):])
        plt.xlim(0,)
        plt.ylim(-120,)
        plt.title('Frequency Response of chosen Window (Convolution Theorem)')
        plt.xlabel('f in Hz')
        plt.ylabel('dB')
        plt.grid()
        plt.show()
    ## apply window:
    windowedData = []
    count = 0
    minDataLength = []
    for i in range(numOfSensors):
        minDataLength.append(np.size(data[0][::,1]))
    minDataLength = np.min(minDataLength)
    numOfWindows = int(minDataLength/windowNumberOfPoints)
    for i in range(numOfWindows):
        if ( (count>=(windowShift-1)) | (windowShift==1) ):
            windowedDataTemp = []
            for j in range(numOfSensors):
                dataTemp = data[j][i:(i+windowNumberOfPoints), 1]
                windowedDataTemp.append(dataTemp * windowFunction)
            windowedDataTemp = np.transpose(windowedDataTemp)
            windowedData.append(windowedDataTemp)
            count = 0
        count += 1
    data = windowedData
    return data, numOfWindows


## version 3
########################################################################################################################
## cb, 10.04.18
##
## general info:
## select data slices (training / test / ...) split data to windows
## new data window each x time steps ...
## no time axis input
## data cant be cut...
##
## function inputs:
## data: data[j,k] -> i-th datapoint, j-th sensor
## windowWidth: width of data windows in number of pointss
## windowShift: after how many time steps a new window starts?
## sampleT: sampling period time in seconds
## enaCheck: enables plotting of chosen slices
## window: choice of window function (https://en.wikipedia.org/wiki/Window_function)
##      - tukey -> tukey window (flattened cosine)
##        rect -> rectangular window
##        bart -> bartlett window
##        black -> blackman window
##        ham -> hamming window
##        hann -> hanning window (raised-cosine window)
##        kaiser -> kaiser window
##        gauss -> gaussian window
## alpha: shape parameter for tukey, gaussian and kaiser window
## enaCWF: show window function and fft
##
## functions output:
## data: data[i][j,k] -> i-th window, j-th data point of k-th sensor
## numOfWindows: number of returned data windows (for convenience)
##
########################################################################################################################
##
def sliceAndWindowV3(data, windowWidth, enaCheck=1, windowShift=1, window='tukey', alpha=0.1, enaCWF=0):
    ## cut slice from input-data:
    numOfSensors = len(data[::].T)
    if enaCheck:
        plt.figure()
        for i in range(numOfSensors):
            plt.plot(data[::, i], label = i)
        plt.legend()
        plt.title('Check selected data!')
        plt.show()
    ## cut slices for windows
    windowNumberOfPoints = windowWidth
    ## choice of window function:
    if (window=='tukey'):
        windowFunction = scipy.signal.tukey(windowNumberOfPoints, alpha)
    elif (window=='rect'):
        windowFunction = np.ones(windowNumberOfPoints)
    elif (window=='bart'):
        windowFunction = np.bartlett(windowNumberOfPoints)
    elif (window=='black'):
        windowFunction = np.blackman(windowNumberOfPoints)
    elif (window=='ham'):
        windowFunction = np.hamming(windowNumberOfPoints)
    elif (window=='hann'):
        windowFunction = np.hanning(windowNumberOfPoints)
    elif (window=='kaiser'):
        windowFunction = np.kaiser(windowNumberOfPoints, alpha)
    elif (window=='gauss'):
        windowFunction = scipy.signal.gaussian(windowNumberOfPoints, alpha)
    if enaCWF==1:
        timeAxis = np.linspace(0,windowWidth, windowNumberOfPoints)
        plt.plot(timeAxis, windowFunction)
        plt.title('Time Function of chosen Window')
        plt.xlabel('t in s')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.figure()
        SAMPLING_FREQUENCY = 1/0.0165 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SAMPLING_FREQUENCY !!!!!!!!!!!!!!!!!!!!
        freqAxis = np.linspace(0,SAMPLING_FREQUENCY/2,int(windowNumberOfPoints))
        windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(np.concatenate( (np.zeros(int(windowNumberOfPoints/2)),windowFunction,np.zeros(int(windowNumberOfPoints/2)) )))))
        windowFreqResponse = 20*np.log10(windowFreqResponse/np.max(windowFreqResponse))
        plt.plot(freqAxis,windowFreqResponse[int(windowNumberOfPoints):])
        plt.xlim(0,)
        plt.ylim(-120,)
        plt.title('Frequency Response of chosen Window (Convolution Theorem)')
        plt.xlabel('f in Hz')
        plt.ylabel('dB')
        plt.grid()
        plt.show()
    ## apply window:
    windowedData = []
    minDataLength = []
    for i in range(numOfSensors):
        minDataLength.append(np.size(data[::,i]))
    minDataLength = np.min(minDataLength)
    numOfWindows = int((minDataLength-windowNumberOfPoints)/windowShift+1)
    for i in range(numOfWindows):
        windowedDataTemp = []
        for j in range(numOfSensors):
            dataTemp = data[i*windowShift:(i*windowShift+windowNumberOfPoints), j]
            windowedDataTemp.append(dataTemp * windowFunction)
        windowedDataTemp = np.transpose(windowedDataTemp)
        windowedData.append(windowedDataTemp)
    data = windowedData
    return data, numOfWindows