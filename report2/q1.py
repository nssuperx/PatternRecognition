# -*- codeing: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def myrand_gmm(n, mu, sigma, fill=0.0):
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    flag = (0 <= u) & (u < 1/2)          # この&は論理積（ビット演算）
    x = (mu[0] + sigma[0]*g)*flag        # この例は、各分布から1/2の確率でデータが出現する例
    flag = (1/2 <= u) & (u < 1)
    x += (mu[1] + sigma[1]*g)*flag
    return x

n = 1000
m = 2
mu = np.array([1.0, 2.0])
sigma = np.array([0.1, 0.3])
sigma2 = sigma * sigma
w = np.ones(m)/m
x = myrand_gmm(n, mu, sigma)

xx = np.arange(0,3.5,0.01)
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y = w[0] * y0 + w[1] * y1

plt.hist(x, bins=100, density=True)
plt.plot(xx, y, color='r', label="q(x;theta)")
plt.plot(xx, y0, color='y', label="sigma1")
plt.plot(xx, y1, color='b', label="sigma2")
plt.legend()
plt.show()
