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
def extractFeatures(dataWindow, numDomCoeffs, numDomFreqs, statFeat=True, wavelet='haar'):
    featureVector = []
    windowNumberOfPoints = np.size(dataWindow[::, 0])
    numOfSensors = np.size(dataWindow[0, ::])
    ## wavelet features:
    dominantCoeffsVal = []
    dominantCoeffsVal1 = []
    dominantCoeffsAmp = []
    dominantCoeffsAmp1 = []
    if numDomCoeffs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]/np.sum(np.square(dataWindow[::, i]))
            coeffs = pywt.wavedec(dataWindow[::, i], wavelet=wavelet, mode='symmetric', level=1)
            coeffs0 = coeffs[0]
            coeffs1 = coeffs[1]
            translationAxis = np.linspace(-1, 1, np.size(coeffs0))
            dominantCoeffsAmp.append(coeffs0[coeffs0.argsort()[-numDomCoeffs:]])
            dominantCoeffsAmp1.append(coeffs0[coeffs1.argsort()[-numDomCoeffs:]])
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
            if (np.max(coeffs1)==0):
                dominantCoeffsVal1.append(np.zeros(numDomCoeffs))
            else:
                dominantCoeffsVal1.append(translationAxis[coeffs1.argsort()[-numDomCoeffs:]])
            for j in range(np.size(dominantCoeffsVal1[i])):
                temp = dominantCoeffsVal1
                featureVector.append(temp[i][j])
            for j in range(np.size(dominantCoeffsAmp1[i])):
                temp = dominantCoeffsAmp1
                featureVector.append(temp[i][j])
    ## fourier features:
    freqAxis = np.linspace(-1, 1, windowNumberOfPoints)
    dominantFreqVal = []
    dominantFreqAmpRe = []
    dominantFreqAmpIm = []
    if numDomFreqs!=0:
        for i in range(numOfSensors):
            dataWindow[::, i] = dataWindow[::, i]
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
    # plt.plot(featureVector)
    # plt.show()
    ## statistical features:
    if statFeat:
        xCorrWavCoeffs = 2
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