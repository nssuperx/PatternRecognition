# -*- codeing: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 参考:test_gmm.py
# 混合数3, muとsigmaは関数内で適当に設定
def q2func_gmm(n, fill=0.0):
    x = np.zeros(n)
    g = np.random.randn(n)
    u = np.random.rand(n)
    mu = np.array([1.0, 3.0, 5.0])
    sigma = np.array([0.1, 0.3, 0.5])
    flag = (0 <= u) & (u < 1/3)         
    x = (mu[0] + sigma[0]*g)*flag
    flag = (1/3 <= u) & (u < 2/3)
    x += (mu[1] + sigma[1]*g)*flag
    flag = (2/3 <= u) & (u <= 1)
    x += (mu[2] + sigma[2]*g)*flag
    return x

# 参考:p118a.py, p118c.py
n = 1000 # 標本数
m = 3 # 混合数
x = q2func_gmm(n)

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

xx = np.arange(0,7,0.01)
y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
y = w[0] * y0 + w[1] * y1 + w[2] * y2

# axs[0,0].hist(x, bins='auto', normed=True)
axs[0,0].hist(x, bins=100, density=True)
axs[0,0].plot(xx, y, color='r')
# axs[0,0].plot(xx, y0, color='g')
# axs[0,0].plot(xx, y1, color='b')
# axs[0,0].plot(xx, y2, color='y')

 
axs[0,1].plot(wt[0], label="w0")
axs[0,1].plot(wt[1], label="w1")
axs[0,1].plot(wt[2], label="w2")
axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('w0, w1, and w2')
axs[0,1].grid(True)
# axs[0,1].xaxis.set_major_locator(MultipleLocator(2)) # 整数で2ずつ
axs[0,1].legend(bbox_to_anchor=(1, 1), loc='upper right')
             
 
axs[1,0].plot(mut[0], label="mu0")
axs[1,0].plot(mut[1], label="mu1")
axs[1,0].plot(mut[2], label="mu2")
axs[1,0].set_xlabel('time')
axs[1,0].set_ylabel('mu0, mu1, and mu2')
axs[1,0].grid(True)
# axs[1,0].xaxis.set_major_locator(MultipleLocator(2))
axs[1,0].legend(bbox_to_anchor=(1, 1), loc='upper right')
             
axs[1,1].plot(np.sqrt(sigma2t[0]), label="sigma0")
axs[1,1].plot(np.sqrt(sigma2t[1]), label="sigma1")
axs[1,1].plot(np.sqrt(sigma2t[2]), label="sigma2")
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel('sigma0, 1, and 2')
axs[1,1].grid(True)
# axs[1,1].xaxis.set_major_locator(MultipleLocator(2))
axs[1,1].legend(bbox_to_anchor=(1, 1), loc='upper right')

fig.tight_layout() # 余白をそろえる
plt.show()