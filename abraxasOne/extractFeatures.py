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
from abraxasOne.plotMatrixWithValues import plotMatrixWithValues
from statsmodels.robust import mad
def extractFeatures(dataWindow, numDomCoeffs, numDomFreqs, wvltLvl1=False, statFeat=True, wavelet='haar'):
    featureVector = []
    windowNumberOfPoints = np.size(dataWindow[::, 0])
    numOfSensors = np.size(dataWindow[0, ::])
    # wavelet features:
    dominantCoeffsVal = []
    dominantCoeffsVal1 = []
    dominantCoeffsAmp = []
    dominantCoeffsAmp1 = []
    if numDomCoeffs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]
            coeffs = pywt.wavedec(dataWindow[::, i], wavelet=wavelet, mode='symmetric', level=1)
            coeffs0 = coeffs[0]
            coeffs1 = coeffs[1]
            dominantCoeffsAmp.append(coeffs0[coeffs0.argsort()[-numDomCoeffs:]])
            dominantCoeffsAmp1.append(coeffs1[coeffs1.argsort()[-numDomCoeffs:]])
            translationAxis = np.linspace(-1, 1, np.size(coeffs0))
            translationAxis1 = np.linspace(-1, 1, np.size(coeffs1))
            #
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
            # first level
            if wvltLvl1:
                if (np.max(coeffs1)==0):
                    dominantCoeffsVal1.append(np.zeros(numDomCoeffs))
                else:
                    dominantCoeffsVal1.append(translationAxis1[coeffs1.argsort()[-numDomCoeffs:]])
                for j in range(np.size(dominantCoeffsVal1[i])):
                    temp = dominantCoeffsVal1
                    featureVector.append(temp[i][j])
                for j in range(np.size(dominantCoeffsAmp1[i])):
                    temp = dominantCoeffsAmp1
                    featureVector.append(temp[i][j])
    # fourier features:
    freqAxis = np.linspace(-1, 1, int(windowNumberOfPoints))
    dominantFreqVal = []
    dominantFreqAmp = []
    dominantFreqPha = []
    if numDomFreqs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]
            spectrum = np.fft.fftshift(np.fft.fft(dataWindow[::, i]))[int(windowNumberOfPoints/2):]
            absSpectrum = np.abs(np.fft.fftshift(np.fft.fft(dataWindow[::, i])))[int(windowNumberOfPoints/2):]
            reS = np.real(spectrum[absSpectrum.argsort()[-numDomFreqs:]])
            imS = np.imag(spectrum[absSpectrum.argsort()[-numDomFreqs:]])
            dominantFreqAmp.append(np.sqrt(reS**2+imS**2))
            dominantFreqPha.append(np.arctan(imS/reS)+0.01*reS)
            dominantFreqVal.append(freqAxis[absSpectrum.argsort()[-numDomFreqs:]])
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
            featureVector.append(np.mean(dataWindow[::, i]))
            featureVector.append(np.var(dataWindow[::, i]))
            featureVector.append(mad(dataWindow[::, i]))
            for j in range(numOfSensors-i-1):
                correlation = np.correlate(dataWindow[::, i], dataWindow[::, j+1], mode='same')\
                             /np.sum(dataWindow[::, i])/np.size(dataWindow[::, i])
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