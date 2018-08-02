import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab


def kernel(x):
    #    return 1/2*np.exp(-np.abs(x))
    #    return 1/(np.pi + x**2)
    try:
        y = 1 / np.sqrt(2 * np.pi) * np.exp( - 1 / 2 * (x ** 2))
    except FloatingPointError:
        y = 0.0
    return y


def kernelDensityEstimator(x, h, ):
    np.seterr(all='raise')

    mu = 0
    sigma = 1

    n = len(x)
    f = []
    t = np.linspace(-5, 5, 500)
    for i in range(len(t)):
        f.append(1/(n*h)*np.sum(kernel((t[i]-x)/h)))

    # plt.plot(t, f)
    # plt.plot(t,mlab.normpdf(t, mu, sigma),'r--')
    # plt.show()
    return f

