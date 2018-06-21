########################################################################################################################
## cb 20.06.2018
##
## general info:
## second version... focus on normalization of features
##
## function inputs:
## dataWindow: dataWindow[j,k] -> j-th data-point, k-th sensor
## numDomCoeffs: number of dominant wavelet coefficients returned
## numDomFreqs: number of dominant frequencies coefficients returned
## wavelet: wavelt to be used
##
## function outputs:
## feature vector with wavelet and fourier features (dominant coefficients)
##
import matplotlib.pyplot as plt
import numpy as np
import pywt
from statsmodels.robust import mad
def extractFeatures(dataWindow, numDomCoeffs, numDomFreqs, statFeat=True, wavelet='haar'):
    featureVector = []
    windowNumberOfPoints = np.size(dataWindow[::, 0])
    numOfSensors = np.size(dataWindow[0, ::])
    ## wavelet features:
    dominantCoeffsVal = []
    dominantCoeffsAmp = []
    if numDomCoeffs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]/np.sum(np.square(dataWindow[::, i]))
            coeffs = pywt.wavedec(dataWindow[::, i], wavelet=wavelet, mode='symmetric', level=0)
            coeffs0 = coeffs[0]
            translationAxis = np.linspace(-1, 1, np.size(coeffs0))
            dominantCoeffsAmp.append(coeffs0[coeffs0.argsort()[-numDomCoeffs:]])
            if (np.max(coeffs0)==0):
                dominantCoeffsVal.append(np.zeros(numDomCoeffs))
            else:
                dominantCoeffsVal.append(translationAxis[coeffs0.argsort()[-numDomCoeffs:]])
            for j in range(np.size(dominantCoeffsVal[i])):
                temp = dominantCoeffsVal
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantCoeffsAmp[i])):
                temp = dominantCoeffsAmp
                featureVector.append(temp[i][j])
    ## fourier features:
    freqAxis = np.linspace(-1, 1, windowNumberOfPoints)
    dominantFreqVal = []
    dominantFreqAmpRe = []
    dominantFreqAmpIm = []
    if numDomFreqs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]/np.sum(np.square(dataWindow[::, i]))
            spectrum = np.fft.fftshift(np.fft.fft(dataWindow[::, i]))[int(windowNumberOfPoints/2):]
            absSpectrum = np.abs(np.fft.fftshift(np.fft.fft(dataWindow[::, i])))[int(windowNumberOfPoints/2):]
            dominantFreqAmpRe.append(np.real(spectrum[absSpectrum.argsort()[-numDomFreqs:]]))
            dominantFreqAmpIm.append(np.imag(spectrum[absSpectrum.argsort()[-numDomFreqs:]]))
            dominantFreqVal.append(freqAxis[absSpectrum.argsort()[-numDomFreqs:]])
            for j in range(np.size(dominantFreqVal[i])):
                temp = dominantFreqVal
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantFreqAmpRe[i])):
                temp = dominantFreqAmpRe
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantFreqAmpIm[i])):
                temp = dominantFreqAmpIm
                featureVector.append(temp[i][j])
    ## statistical features:
    if statFeat:
        for i in range(numOfSensors):
            featureVector.append(np.mean(dataWindow[::, i]))
            featureVector.append(np.var(dataWindow[::, i]))
            featureVector.append(mad(dataWindow[::, i]))
    featureVector = np.reshape(featureVector, np.size(featureVector))
    return featureVector