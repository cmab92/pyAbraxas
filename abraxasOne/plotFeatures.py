import matplotlib.pyplot as plt
import numpy as np
import pywt
from statsmodels.robust import mad
from mpl_toolkits.mplot3d import Axes3D

from abraxasOne.loadAndClean import loadAndClean
from abraxasOne import helperFunctions as hf
from abraxasOne.helperFunctions import writeDataToCsvFile
from abraxasOne.gaussFilter import gaussFilter
from abraxasOne.sliceAndWindow import sliceAndWindowV3
from abraxasOne.extractSpectralFeatures import extractSpectralFeatures


dataWindow = np.array(data[o])
        wavelet = 'haar'
        coeffNormFact = 1
        numDomCoeffs = 6
        sampleT = 0.0165
        numDomFreqs = 7
        freqNormFact = 1



        numOfSensors = np.size(dataWindow[0, ::])
        ## wavelet features:
        WAVEMODE = 'symmetric'  # zero padding
        THRESMODE = 'hard'  # soft thresholdin
        windowNumberOfPoints = np.size(dataWindow[::, 0])
        windowCoeffs = []
        # waveletDenoised = []
        dominantCoeffsVal = []
        dominantCoeffsAmp = []
        signalPower = []

        waveletDenoised = []
        waveFeat = []
        freqFeat = []


        for i in range(10):
            coeffs = pywt.wavedec(data[o][::,i], wavelet=wavelet, mode=WAVEMODE, level=0)
            coeffs0 = np.abs(coeffs[0])
            coeffs0_ = np.zeros(np.shape(coeffs))
            dominantCoeffsAmp.append(coeffs0[coeffs0.argsort()[-numDomCoeffs:]])
            domCoeff=coeffs0[coeffs0.argsort()[-numDomCoeffs:]]
            for j in range(numDomCoeffs):
                for k in range(np.size(coeffs)):
                    if np.abs(coeffs[0][k])==domCoeff[j]:
                        coeffs0_[0][k] = coeffs[0][k]
            for j in range(np.size(coeffs)):
                coeffs[0][j] = coeffs0_[0][j]

            waveFeat.append(pywt.waverec(coeffs, wavelet=wavelet, mode=WAVEMODE))
            #plt.show()

        # windowSpectrum = []
        dominantFreqVal = []
        dominantFreqAmp = []
        for i in range(10):
            orig_spec = np.fft.fft(dataWindow[::, i])
            orig_spec_Temp = np.zeros(np.shape(orig_spec))
            spectrum = np.abs(np.fft.fft(dataWindow[::, i]))
            domFreq = spectrum[spectrum.argsort()[-numDomFreqs:]]
            #print(domFreq)
            for j in range(numDomFreqs):
                for k in range(np.size(orig_spec)):
                    if np.abs(orig_spec[k])==domFreq[j]:
                        orig_spec_Temp[k] = orig_spec[k]
            orig_spec = orig_spec_Temp
            freqFeat.append(np.fft.ifft(orig_spec))