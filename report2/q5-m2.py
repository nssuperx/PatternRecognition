# -*- codeing: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def myrand(n, fill=0.0):
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


# 参考:p118a.py, p118c.py
n = 10000 # 標本数
m = 2 # 混合数
x = myrand(n)

L = -np.inf
w = np.ones(m)/m
w = w.reshape(m,1) # 縦ベクトルにする
mu = np.linspace( min(x), max(x), m)   # 平均値の初期値
mu = mu.reshape(m,1)
sigma2 = np.ones(m)/10  # 分散の初期値
sigma2 = sigma2.reshape(m,1)

Lt = L
wt = w
mut = mu
sigma2t = sigma2
t=0
tt = np.array([0]) 

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

    wt = np.append(wt,w, axis=1)
    mut = np.append(mut,mu, axis=1)
    sigma2t = np.append(sigma2t,sigma2, axis=1)
    # print(sigma2t)

    if Lnew - L < 0.0001:
        break
    L = Lnew

    Lt = np.append(Lt,L)
    t = t+1
    tt = np.append(tt,t)

# http://bicycle1885.hatenablog.com/entry/2014/02/14/023734
# https://qiita.com/Tatejimaru137/items/50fb90dd52f194979a13
fig, axs = plt.subplots(2, 2)

xx = np.arange(0,6,0.01)
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y = w[0] * y0 + w[1] * y1

# axs[0,0].hist(x, bins='auto', normed=True)
axs[0,0].hist(x, bins=100, density=True)
axs[0,0].plot(xx, y, color='r')
 
axs[0,1].plot(wt[0], label="w0")
axs[0,1].plot(wt[1], label="w1")
axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('w0, w1, and w2')
axs[0,1].grid(True)
# axs[0,1].xaxis.set_major_locator(MultipleLocator(2)) # 整数で2ずつ
axs[0,1].legend(bbox_to_anchor=(1, 1), loc='upper right')
             
 
axs[1,0].plot(mut[0], label="mu0")
axs[1,0].plot(mut[1], label="mu1")
axs[1,0].set_xlabel('time')
axs[1,0].set_ylabel('mu0, mu1, and mu2')
axs[1,0].grid(True)
# axs[1,0].xaxis.set_major_locator(MultipleLocator(2))
axs[1,0].legend(bbox_to_anchor=(1, 1), loc='upper right')
             
axs[1,1].plot(sigma2t[0], label="sigma0")
axs[1,1].plot(sigma2t[1], label="sigma1")
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel('sigma0, 1, and 2')
axs[1,1].grid(True)
# axs[1,1].xaxis.set_major_locator(MultipleLocator(2))
axs[1,1].legend(bbox_to_anchor=(1, 1), loc='upper right')

fig.tight_layout() # 余白をそろえる
plt.show()