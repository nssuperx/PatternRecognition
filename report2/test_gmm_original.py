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

def myrand(n, fill=0.0): # n は生成するデータの個数
    x=np.zeros(n)
    u=np.random.rand(n)  
    flag=(0<=u)  &  (u<1/8)
    x = np.sqrt(8*u)*flag
    flag=(1/8<=u)  &  (u<1/4)
    x += ( 2-np.sqrt( (2-8*u)*flag )    )*flag 
    flag=(1/4<=u)  &  (u<1/2)
    x += (1+4*u)*flag
    flag=(1/2<=u)  &  (u<3/4)
    x += (3+np.sqrt(   (4*u-2)*flag ) )*flag
    flag=(3/4<=u)  &  (u<=1)
    x += (5-np.sqrt(4-4*u) )*flag
    return x

n = 10000
x = myrand(n)
m = 3
mu = np.array([1.0, 2.0, 3.0])
sigma = np.array([0.1, 0.3, 0.5])
sigma2 = sigma * sigma
w = np.ones(m)/m

xx = np.arange(0,5,0.01)
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
y = w[0] * y0 + w[1] * y1 + w[2] * y2

plt.plot(xx, y, color='r')
plt.hist(x, bins=100, density=True)
plt.show()
