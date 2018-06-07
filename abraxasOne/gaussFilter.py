########################################################################################################################
## gauss filter in frequency domain
## todo: what about Re(F) and Im(F) ???
########################################################################################################################
import numpy as np
def gaussFilter(x=1, y=1, AMP=1, MEAN=0, SIGMA=0.5):
    ## compute spectrum:
    dataSpectrum = np.array( np.fft.fft(y) )
    ## compute frequency axis:
    if np.size(x)<2:
        FSAMPLE = 2 # f_s/2 = 1
    else:
        FSAMPLE = 1 / (np.sum(np.diff(np.flip(-x, 0))) / np.size(x)) # 1/average(T_sample)
    freqAxis = np.array( np.linspace(-FSAMPLE/2,FSAMPLE/2,np.size(dataSpectrum)) )
    ## compute filtered spectrum:
    filteredSpectrum = np.array(np.fft.ifftshift(np.fft.fftshift(dataSpectrum)*(AMP*np.exp(-(freqAxis-MEAN)**2/(2*SIGMA**2))/np.size(freqAxis))))
    filteredSpectrum = filteredSpectrum*dataSpectrum[0]/filteredSpectrum[0]
    ## compute filtered data:
    filteredData = np.real( (np.fft.ifft(filteredSpectrum)) )
    return filteredData