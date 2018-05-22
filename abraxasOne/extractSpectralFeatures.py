########################################################################################################################
## cb, 05.04.18
##
## general info:
## for fourier and wavelet transfrom the 'spectral-axis' is normlaized to 1
##
## function inputs:
## dataWindow: dataWindow[j,k] -> j-th data-point, k-th sensor
## numDomCoeffs: number of dominant wavelet coefficients returned
## numDomFreqs: number of dominant frequencies coefficients returned
## sampleT: sampling time in seconds
##
## functions output:
## feature vector with wavelet and fourier features (dominant coefficients)
##
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.signal
import scipy.ndimage
from statsmodels.robust import mad
from sklearn import svm
from helperFunctions import scalogram
##
## numDomCoeffs: number of dominant wavelet coefficients to be selected <- todo: optimize
## numDomFreqs: number of dominant frequencies to be selected <- todo: optimize
def extractSpectralFeatures(dataWindow, numDomCoeffs=2, numDomFreqs=2, sampleT=0.012, wavelet = 'haar', coeffNormFact=1, freqNormFact=1):
    featureVector = []
    numOfSensors = np.size(dataWindow[0,::])
    ## wavelet features:
    LEVEL = 0
    WAVEMODE = 'symmetric'  # zero padding
    THRESMODE = 'hard'  # soft thresholdin
    windowNumberOfPoints = np.size(dataWindow[::, 0])
    windowCoeffs = []
    #waveletDenoised = []
    dominantCoeffsVal = []
    dominantCoeffsAmp = []
    signalPower = []
    for i in range(numOfSensors):
        signalPower.append(np.sum(np.abs(dataWindow[::,i])**2))
        coeffs = pywt.wavedec(dataWindow[::, i], wavelet=wavelet, mode=WAVEMODE, level=LEVEL)
        for j in range(LEVEL):
            coeffs[j] = coeffs[j]*coeffNormFact/signalPower[i]/windowNumberOfPoints
            thresh = mad(coeffs[j]) * np.sqrt(2 * np.log(len(dataWindow[::, i])))
            coeffs[j] = pywt.threshold(coeffs[j], value=thresh, mode=THRESMODE)
        #waveletDenoised.append(pywt.waverec(coeffs, wavelet=WAVELET, mode=WAVEMODE))
        windowCoeffs.append(coeffs[:LEVEL])  # omit last level
        coeffs0 = np.abs(coeffs[0])
        translationAxis = np.linspace(0, 1, np.size(coeffs0))
        dominantCoeffsAmp.append(coeffs0[coeffs0.argsort()[-numDomCoeffs:]])
        if (np.max(coeffs0)==0):
            dominantCoeffsVal.append(np.zeros(numDomCoeffs))
        else:
            dominantCoeffsVal.append(translationAxis[coeffs0.argsort()[-numDomCoeffs:]])
    for i in range(numDomCoeffs):
        for j in range(numOfSensors):
            featureVector.append(np.round([(np.transpose(dominantCoeffsVal)[i,j], np.transpose(dominantCoeffsAmp)[i,j])],5))
    ## fourier features:
    freqAxis = np.linspace(-(2/sampleT), 2/sampleT, windowNumberOfPoints)
    freqAxis = freqAxis[int(windowNumberOfPoints / 2):]*sampleT/2 # from zero to nyquist frequency but normalized to 1

    #windowSpectrum = []
    dominantFreqVal = []
    dominantFreqAmp = []
    for i in range(numOfSensors):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(dataWindow[::, i])))
        spectrum = spectrum[int(windowNumberOfPoints/2):]*freqNormFact/signalPower[i]/windowNumberOfPoints
    #windowSpectrum.append(spectrum)
        dominantFreqAmp.append(spectrum[spectrum.argsort()[-numDomFreqs:]]) # -> get amplitude of largest frequencies
        dominantFreqVal.append(freqAxis[spectrum.argsort()[-numDomFreqs:]]) # -> get value of largest frequencies
    for i in range(numDomFreqs):
        for j in range(numOfSensors):
            featureVector.append(np.round([(np.transpose(dominantFreqVal)[i,j], np.transpose(dominantFreqAmp)[i,j])],5))
    featureVector = np.reshape(featureVector, np.size(featureVector))
    #plt.plot(featureVector)
    return featureVector