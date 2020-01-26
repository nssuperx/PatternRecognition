# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
 
def myrand_gmm(n, fill=0.0): # n は生成するデータの個数
  x=np.zeros(n)
  g=np.random.randn(n)
  u=np.random.rand(n)
  mu = np.array([1.0, 2.0, 3.0])     # 各ガウス分布の平均値．値を変えていろいろ試す． これは混合数 m=3 nの場合
  sigma = np.array([0.1, 0.3, 0.8]) # 各分布の標準偏差．値を変えていろいろ試す．
  flag=(0<=u)  &  (u<1/3) # この例は，各分布から1/3の確率でデータが出現する場合．
  x = (mu[0] + sigma[0]*g)*flag
  flag=(1/3<=u)  &  (u<2/3)
  x += (mu[1] + sigma[1]*g)*flag
  flag=(2/3<=u)  &  (u<=1)
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
 
 
# Fixing random state for reproducibility
np.random.seed(20190105)
n = 100000 # 標本数（サンプル数）．
# x = myrand_gmm(n) # 乱数の生成．実験に使うデータを生成する関数名に書き換える．
x = myrand(n) 
m = 3  # 混合数．この値を変えて実験する．
  
# 初期値の設定．w, mu, sigma2 は m 次元縦ベクトル．
L = -np.inf
w = np.ones(m)/m # m個の正規分布の重みの初期値
w = w.reshape(m,1)
mu = np.linspace( min(x), max(x), m)   # 平均値の初期値
mu = mu.reshape(m,1) # m を明示的に縦ベクトルにする
sigma2 = np.ones(m)/10  # 分散の初期値
sigma2 = sigma2.reshape(m,1)
 
# mu_t = np.empty((1,m), float)
 
Lt = L
wt = w
mut = mu
sigma2t = sigma2
t=0
tt = np.array([0]) 
 
while 1:
  tmp1 = ( np.tile(x, (m,1)) - np.tile(mu, (1, n)) )* ( np.tile(x, (m,1)) - np.tile(mu, (1, n)) )
  tmp2 = 2*np.tile( sigma2, (1, n) )
  tmp3 = np.tile(w, (1, n) )*np.exp(-tmp1/tmp2)/np.sqrt(np.pi*tmp2)
  eta = tmp3/np.tile(np.sum(tmp3, axis=0), (m, 1) )  # ここまでがηの計算
  tmp4 = np.sum(eta, axis=1)
  w = tmp4/n
  w= w.reshape(m,1)
  mu = (eta.dot(x))/tmp4
  mu = mu.reshape(m,1)
   
  sigma2 = np.sum(tmp1*eta, axis=1)/tmp4
  sigma2 = sigma2.reshape(m,1)
  Lnew = np.sum(np.log(np.sum(tmp3,axis=0))) # 更新後の対数尤度
 
  wt = np.append(wt,w, axis=1)
  mut = np.append(mut,mu, axis=1)
  sigma2t = np.append(sigma2t,sigma2, axis=1)
   
  # mu, sigma2, w
   
  if Lnew -L < 0.0001:
    break
  L = Lnew
  Lt = np.append(Lt,L)
  t = t+1
  tt = np.append(tt,t)
     
xx = np.arange(0,5,0.01) #0から5まで、0.01間隔．
y0 = norm.pdf(xx,  mu[0], np.sqrt(sigma2[0] ) )
y1 = norm.pdf(xx,  mu[1], np.sqrt(sigma2[1] ) )
y2 = norm.pdf(xx,  mu[2], np.sqrt(sigma2[2] ) )
y = w[0]*y0 + w[1]*y1 + w[2]* y2
 
 
fig, axs = plt.subplots(2, 2)
 
#plt.plot(xx,y,color='r')
# plt.hist(x, bins='auto', normed=True)
#plt.hist(x, bins=50, normed=True)
 
axs[0,0].plot(xx,y,color='r')
axs[0,0].hist(x, bins='auto', normed=True)
 
axs[0,1].plot(wt[0])
axs[0,1].plot(wt[1])
axs[0,1].plot(wt[2])
# axs[0].set_xlim(0, 2)
axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('w0, w1, and w2')
axs[0,1].grid(True)
             
 
axs[1,0].plot(mut[0])
axs[1,0].plot(mut[1])
axs[1,0].plot(mut[2])
# axs[0].set_xlim(0, 2)
axs[1,0].set_xlabel('time')
axs[1,0].set_ylabel('mu0, mu1, and mu2')
axs[1,0].grid(True)
             
axs[1,1].plot(sigma2t[0])
axs[1,1].plot(sigma2t[1])
axs[1,1].plot(sigma2t[2])
# axs[0].set_xlim(0, 2)
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel('sigma0, 1, and 2')
axs[1,1].grid(True)
 
#plt.plot(Lt)
 
#axs[3].set_xlabel('time')
##axs[3].set_ylabel('Likelihood')
#axs[3].grid(True)
#axs[3].plot(Lt)
 
fig.tight_layout()
plt.show()