"""
cb, 10.07.2018

 - feature normalization

 Details:
 Normalize a matrix of feature vectors and return normalization parameters or normalize single feature vector, given
 a set of normalization parameters.

Input:
features    := Single feature vector or feature matrix. Feature matrix has to be of shape: features[i, j], where i
            refers to the i-th feature vector and j to the j-th feature.

method      := Normalization method:
            stand   -> standardization, x_new = (x - mean(x))/standard_dev(x)
            mean    -> mean normalization, x_new = (x - mean(x))/(max(x) - min(x))
            minmax  -> rescaling, (x - min(x))/(max(x) - min(x)).
            Default stand.

param       := Matrix with normalization parameters. If not given, the parameters are chosen with respect to the
            given features.

Output:
features    := Normalized features.

normVal     := Normalization parameters. Either mean and standart deviation, or mean, min and max, or min and max. '
            Only given, if no param input.

"""

import numpy as np


def normalize(features, method='stand', param=None):
    if param is None:
        if np.size(features) == len(features):
            print("Single feature vector for normalization!?")
        if method == 'stand':
            mue = []
            sigma = []
            for i in range(len(features[0][::])):
                mue.append(np.mean(features[::, i]))
                sigma.append(np.std(features[::, i]))
                if sigma[i] == 0:
                    sigma[i] = np.mean(features[0, i])*10**6 + 10**6
                features[::, i] = (features[::, i] - mue[i])/sigma[i]
            normVal = [mue, sigma]
        elif method == 'mean':
            mue = []
            minVal = []
            maxVal = []
            for i in range(len(features[0][::])):
                mue.append(np.mean(features[::, i]))
                minVal.append(np.min(features[::, i]))
                maxVal.append(np.max(features[::, i]))
                if (maxVal[i] - minVal[i]) == 0:
                    maxVal[i] = np.abs(minVal[i]*10**6)
                features[::, i] = (features[::, i] - mue[i])/(maxVal[i] - minVal[i])
            normVal = [mue, minVal, maxVal]
        elif method == 'minmax':
            minVal = []
            maxVal = []
            for i in range(len(features[0][::])):
                minVal.append(np.min(features[::, i]))
                maxVal.append(np.max(features[::, i]))
                if (maxVal[i] - minVal[i]) == 0:
                    maxVal[i] = np.abs(minVal[i]*10**6)
                features[::, i] = (features[::, i] - minVal[i])/(maxVal[i] - minVal[i])
            normVal = [minVal, maxVal]
        else:
            mue = []
            sigma = []
            for i in range(len(features[0][::])):
                mue.append(np.mean(features[::, i]))
                sigma.append(np.std(features[::, i]))
                if sigma[i] == 0:
                    sigma[i] = np.mean(features[0, i])*10**1 + 10**6
                features[::, i] = (features[::, i] - mue[i])/sigma[i]
            normVal = [mue, sigma]
        normVal = np.array(normVal)
        return features, normVal
    else:
        param = np.array(param).T
        if method == 'stand':
            mue = param[::, 0]
            sigma = param[::, 1]
            features = (features - mue)/sigma
        elif method == 'mean':
            mue = param[::, 0]
            minVal = param[::, 1]
            maxVal = param[::, 2]
            features = (features - mue)/(maxVal - minVal)
        elif method == 'minmax':
            minVal = param[::, 0]
            maxVal = param[::, 1]
            features = (features - minVal)/(maxVal - minVal)
        else:
            mue = param[::, 0]
            sigma = param[::, 1]
            features = (features - mue)/sigma
        return features
