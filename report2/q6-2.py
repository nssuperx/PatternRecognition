# -*- codeing: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 参考:test_gmm.py
# 混合数3, muとsigmaは関数内で適当に設定
def q2func_gmm(n, fill=0.0):
    x = np.zeros((2,n))
    g = np.random.randn(2,n)
    u = np.random.rand(2,n)
    mu = np.array([[1.0, 3.0, 2.0],[2.0, 1.0, 3.0]])
    sigma = np.array([[0.1, 0.3, 0.2],[0.7, 0.3, 0.1]])
    flag = (0 <= u[0]) & (u[0] < 1/3)       
    x[0] = (mu[0][0] + sigma[0][0]*g[0])*flag
    flag = (1/3 <= u[0]) & (u[0] < 2/3)
    x[0] += (mu[0][1] + sigma[0][1]*g[0])*flag
    flag = (2/3 <= u[0]) & (u[0] <= 1)
    x[0] += (mu[0][2] + sigma[0][2]*g[0])*flag
    flag = (0 <= u[1]) & (u[1] < 1/3)         
    x[1] = (mu[1][0] + sigma[1][0]*g[1])*flag
    flag = (1/3 <= u[1]) & (u[1] < 2/3)
    x[1] += (mu[1][1] + sigma[1][1]*g[1])*flag
    flag = (2/3 <= u[1]) & (u[1] <= 1)
    x[1] += (mu[1][2] + sigma[1][2]*g[1])*flag
    return x

# 参考:p118a.py, p118c.py
n = 1000 # 標本数
m = 3 # 混合数
x = q2func_gmm(n)
#plt.scatter(x[0],x[1])
#plt.show()

plt.scatter(x[0],x[1])

L = -np.inf
w = np.ones((m,1))/m
w = w.reshape(m,1) # 縦ベクトルにする
mu = np.linspace( min(x[0]), max(x[0]), m)   # 平均値の初期値
mu = mu.reshape(m,1)
sigma2 = np.ones(m)/10  # 分散の初期値
sigma2 = sigma2.reshape(m,1)

wt = w
mut = mu
sigma2t = sigma2

xx = np.arange(0,5,0.01)
y = [0,0]

while 1:
    tmp1 = np.square(np.tile(x[0], (m,1)) - np.tile(mu, (1,n)))
    tmp2 = 2 * np.tile(sigma2, (1,n))
    tmp3 = np.tile(w, (1,n)) * np.exp(-tmp1 / tmp2) / np.sqrt(np.pi * tmp2)
    eta = tmp3 / np.tile(np.sum(tmp3, axis=0), (m,1))
    tmp4 = np.sum(eta, axis=1)
    w = tmp4 / n
    w = w.reshape(m, 1)
    mu = (eta.dot(x[0])) / tmp4
    mu = mu.reshape(m, 1)
    sigma2 = np.sum(tmp1*eta, axis=1) / tmp4
    sigma2 = sigma2.reshape(m,1)

    Lnew = np.sum(np.log(np.sum(tmp3,axis=0)))

    
    # グラフ描画
    y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
    y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
    y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
    y[0] = w[0] * y0 + w[1] * y1 + w[2] * y2
    plt.plot(xx, y[0], alpha=0.3, color="r")
    
    wt = np.append(wt,w, axis=1)
    mut = np.append(mut,mu, axis=1)
    sigma2t = np.append(sigma2t,sigma2, axis=1)
    # print(sigma2t)

    if Lnew - L < 0.0001:
        break
    L = Lnew

L = -np.inf
w = np.ones((m,1))/m
w = w.reshape(m,1) # 縦ベクトルにする
mu = np.linspace( min(x[1]), max(x[1]), m)   # 平均値の初期値
mu = mu.reshape(m,1)
sigma2 = np.ones(m)/10  # 分散の初期値
sigma2 = sigma2.reshape(m,1)

wt = w
mut = mu
sigma2t = sigma2


while 1:
    tmp1 = np.square(np.tile(x[1], (m,1)) - np.tile(mu, (1,n)))
    tmp2 = 2 * np.tile(sigma2, (1,n))
    tmp3 = np.tile(w, (1,n)) * np.exp(-tmp1 / tmp2) / np.sqrt(np.pi * tmp2)
    eta = tmp3 / np.tile(np.sum(tmp3, axis=0), (m,1))
    tmp4 = np.sum(eta, axis=1)
    w = tmp4 / n
    w = w.reshape(m, 1)
    mu = (eta.dot(x[1])) / tmp4
    mu = mu.reshape(m, 1)
    sigma2 = np.sum(tmp1*eta, axis=1) / tmp4
    sigma2 = sigma2.reshape(m,1)

    Lnew = np.sum(np.log(np.sum(tmp3,axis=0)))
    
    # グラフ描画
    y0 = norm.pdf(xx, mu[0], np.sqrt(sigma2[0]))    # probability density function
    y1 = norm.pdf(xx, mu[1], np.sqrt(sigma2[1]))
    y2 = norm.pdf(xx, mu[2], np.sqrt(sigma2[2]))
    y[1] = w[0] * y0 + w[1] * y1 + w[2] * y2
    plt.plot(y[1], xx, alpha=0.3, color="m")
    
    
    wt = np.append(wt,w, axis=1)
    mut = np.append(mut,mu, axis=1)
    sigma2t = np.append(sigma2t,sigma2, axis=1)
    # print(sigma2t)

    if Lnew - L < 0.0001:
        break
    L = Lnew

# plt.scatter(y[0],y[1])
#plt.plot(xx, y[0], color='r')
#plt.plot(y[1], xx, color='m')
plt.show()
