"""
cb, 07.07.2018

 - calculate features of given data window

Inputs:
dataWindow      := Input data, given as: data[j,k] j-th data-point, k-th sensor.

numCoeffs =     := Number of dominant wavelet coefficients appended to the feature vector.

numFreqs        := Number of dominant frequencies appended to the feature vector.

wvltLvl1        := Add second level (lvl0 and lvl1) coefficients to feature vector. Default False.

statFeat        := Enables statistical features (mean, variance, mad, and peaks of cross-correlation). Default True.

wavelet         := Wavelet used for dwt. Default 'haar'.

dataQueue       := Set true if dataWindow is a queue object. Default False.

Output:
featureVector   := Unscaled feature vector.

"""
import numpy as np
import pywt
from statsmodels.robust import mad


def extractFeatures(dataWindow, numCoeffs, numFreqs, wvltLvl1=False, statFeat=True, wavelet='haar', dataQueue=False):
    if dataQueue is True:
        try:
            data = dataWindow.get()
        except AttributeError:
            print("\n \n 'dataOutQueue' in function 'receiveData' is not a queue object...\n \n")
            raise
    else:
        data = dataWindow

    featureVector = []
    windowNumberOfPoints = np.size(data[::, 0])
    numOfSensors = np.size(data[0, ::])
    # wavelet features:
    coeffsVal = []
    coeffsVal1 = []
    coeffsAmp = []
    coeffsAmp1 = []
    if numCoeffs != 0:
        for i in range(numOfSensors):
            data[::, i] = data[::, i]
            coeffs = pywt.wavedec(data[::, i], wavelet=wavelet, mode='symmetric', level=1)
            coeffs0 = coeffs[0]
            coeffs1 = coeffs[1]
            coeffsAmp.append(coeffs0[coeffs0.argsort()[-numCoeffs:]])
            coeffsAmp1.append(coeffs1[coeffs1.argsort()[-numCoeffs:]])
            translationAxis = np.linspace(-1, 1, np.size(coeffs0))
            translationAxis1 = np.linspace(-1, 1, np.size(coeffs1))
            #
            if np.max(coeffs0) == 0:
                coeffsVal.append(np.zeros(numCoeffs))
            else:
                coeffsVal.append(translationAxis[coeffs0.argsort()[-numCoeffs:]])
            for j in range(np.size(coeffsVal[i])):
                temp = coeffsVal
                featureVector.append(temp[i][j])
            for j in range(np.size(coeffsAmp[i])):
                temp = coeffsAmp
                featureVector.append(temp[i][j])
            # first level
            if wvltLvl1:
                if np.max(coeffs1) == 0:
                    coeffsVal1.append(np.zeros(numCoeffs))
                else:
                    coeffsVal1.append(translationAxis1[coeffs1.argsort()[-numCoeffs:]])
                for j in range(np.size(coeffsVal1[i])):
                    temp = coeffsVal1
                    featureVector.append(temp[i][j])
                for j in range(np.size(coeffsAmp1[i])):
                    temp = coeffsAmp1
                    featureVector.append(temp[i][j])
    # fourier features:
    freqAxis = np.linspace(-1, 1, int(windowNumberOfPoints))
    dominantFreqVal = []
    dominantFreqAmp = []
    dominantFreqPha = []
    if numFreqs!=0:
        for i in range(numOfSensors):
            data[::, i] = data[::, i]
            spectrum = np.fft.fftshift(np.fft.fft(data[::, i]))[int(windowNumberOfPoints/2):]
            absSpectrum = np.abs(np.fft.fftshift(np.fft.fft(data[::, i])))[int(windowNumberOfPoints/2):]
            reS = np.real(spectrum[absSpectrum.argsort()[-numFreqs:]])
            imS = np.imag(spectrum[absSpectrum.argsort()[-numFreqs:]])
            dominantFreqAmp.append(np.sqrt(reS**2+imS**2))
            dominantFreqPha.append(np.arctan(imS/reS)+0.01*reS)
            dominantFreqVal.append(freqAxis[absSpectrum.argsort()[-numFreqs:]])
            for j in range(np.size(dominantFreqVal[i])-1):
                temp = dominantFreqVal
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantFreqAmp[i])):
                temp = dominantFreqAmp
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantFreqPha[i])):
                temp = dominantFreqPha
                featureVector.append(temp[i][j])
    # statistical features
    if statFeat:
        xCorrWavCoeffs = 5
        for i in range(numOfSensors):
            featureVector.append(np.mean(data[::, i]))
            featureVector.append(np.var(data[::, i]))
            featureVector.append(mad(data[::, i]))
            for j in range(numOfSensors-i-1):
                correlation = np.correlate(data[::, i], data[::, j+1], mode='same')\
                             /np.sum(data[::, i])/np.size(data[::, i])
                coeffs = pywt.wavedec(correlation, wavelet=wavelet, mode='symmetric', level=0)
                coeffs0 = coeffs[0]
                translationAxis = np.linspace(-1, 1, np.size(coeffs0))
                domCorrCoeffsAmp = coeffs0[coeffs0.argsort()[-xCorrWavCoeffs:]]
                if np.max(coeffs0) == 0:
                    domCorrCoeffsVal = np.zeros(xCorrWavCoeffs)
                else:
                    domCorrCoeffsVal = translationAxis[coeffs0.argsort()[-xCorrWavCoeffs:]]
                for k in range(xCorrWavCoeffs):
                    featureVector.append(domCorrCoeffsVal[k])
                for k in range(xCorrWavCoeffs):
                    featureVector.append(domCorrCoeffsAmp[k])
    featureVector = np.reshape(featureVector, np.size(featureVector))
    return featureVector

