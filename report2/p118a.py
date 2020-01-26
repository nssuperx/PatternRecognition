# -*- codeing: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def myrand_gmm(n, fill=0.0):
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.3, 0.5])
    flag = (0 <= u) & (u < 1/3)
    x = (mu[0] + sigma[0]*g)*flag
    flag = (1/3 <= u) & (u < 2/3)
    x += (mu[1] + sigma[1]*g)*flag
    flag = (2/3 <= u) & (u <= 1)
    x += (mu[2] + sigma[2]*g)*flag

    return x

n = 1000
x = myrand_gmm(n)
m = 3
L = -np.inf
w = np.ones(m)/m
w = w.reshape(m, 1)                  #縦ベクトルにする
mu = np.linspace(min(x), max(x), m)
mu = mu.reshape(m, 1)
sigma2 = np.ones(m)/10
sigma2 = sigma2.reshape(m,1)

while 1:
    tmp1 = np.square(np.tile(x, (m,1)) - np.tile(mu, (1,n)))
    tmp2 = 2 * np.tile(sigma2, (1,n))
    tmp3 = np.tile(w, (1,n)) * np.exp(-tmp1 / tmp2) / np.sqrt(np.pi * tmp2)
    eta = tmp3 / np.tile(np.sum(tmp3, axis=0), (m,1))
    tmp4 = np.sum(eta, axis=1)
    w = tmp4 / n
    w = w.reshape(m, 1)
    mu = (eta.dot(x)) / tmp4
    mu = mu.reshape(m, 1)
    sigma2 = np.sum(tmp1*eta, axis=1) / tmp4
    sigma2 = sigma2.reshape(m,1)
    Lnew = np.sum(np.log(np.sum(tmp3,axis=0)))

    if Lnew - L < 0.0001:
        break
    L = Lnew

xx = np.arange(0,5,0.01)
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
y = w[0] * y0 + w[1] * y1 + w[2] * y2

plt.plot(xx, y, color='r')
plt.hist(x, bins='auto', density=True)
plt.show()