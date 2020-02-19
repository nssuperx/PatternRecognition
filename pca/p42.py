# -*- coding: utf-8 -*-
# google colab を使う場合は，下の2行をコメントアウト
# from google.colab import files
# f= files.upload()

import numpy as np
import scipy.io
data = scipy.io.loadmat("digit.mat")
type(data) # dict
data.keys() # dict_keys(['__header__', '__version__', '__globals__', 'X', 'T'])

type(data["X"]) # numpy.ndarray
x = data["X"]
type(x) # numpy.ndarray
x.shape # (256, 500, 10)
[d, n, nc] = x.shape

z = x.reshape(d, n*nc)
print(z.shape) # (256, 5000)
print(z)

# 分散・共分散行列 V の計算
V = np.cov(z)
V.shape # (256, 256)

# 正定値対称行列 V の固有ベクトル・固有値の計算
[eigval, eigvec] = np.linalg.eig(V)
eigvec.shape # (256, 256)
eigval.shape # (256,)

# ここで固有ベクトルを固有値の大きい順に並べ替える．
index = np.argsort(eigval)[::-1]
eigvec = eigvec[:,index]
eigvec.shape # (256, 256)
e=eigvec[:,0:2] #最初の2つだけとってきてる
e.shape # (256, 2)

X1 = x[:,:,0].T  # 数字1の500例．X1は 500x256 行列 
X1.shape # (500, 256)
C1 = X1.dot(e)  # 第1,2主成分方向の座標，500例．C1は 500x2 行列

X2 = x[:,:,1].T  # 数字2の500例．X2は 500x256 行列 
X2.shape # (500, 256)
C2 = X2.dot(e)  # 第1,2主成分方向の座標，500例．C2は 500x2 行列
C2.shape # (500, 2)

import matplotlib.pyplot as plt


fig = plt.figure()
# fig.patch.set_facecolor('silver') # 背景をシルバー

# plt.subplot(1, 5, 1)
'''
# プロット
plt.scatter(C1[:,0],C1[:,1],s=10, c="red",label="digit 1")
plt.scatter(C2[:,0],C2[:,1],s=10, c="blue",label="digit 2")

# 凡例の表示
plt.legend()

# 描画した内容を画面表示
plt.show()
'''

'''
# ここは元の数字の画像を表示
plt.subplot(1, 5, 2)
X3 = x[:,:,2].T  # 数字3の500例．X3は 500x256 行列
img = np.reshape(X3[0,:],(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)
'''

# 主成分を表示、あとでここをきれいに表示
'''
plt.subplot(2, 5, 1)
e1=eigvec[:,0] # 第1主成分
print(e1.shape) # (256,1)
img = np.reshape(e1,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(2, 5, 2)
e1=eigvec[:,1] # 第1主成分
print(e1.shape) # (256,1)
img = np.reshape(e1,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)
'''
# 問題1のコード
'''
for i in range(10):
    plt.subplot(2,5,i+1)
    eig = eigvec[:,i]
    img = np.reshape(eig,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)
'''

# 問題2のコード
'''
plt.subplot(1, 3, 1)
e50=eigvec[:,49] # 第50主成分
img = np.reshape(e50,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 3, 2)
e100=eigvec[:,99] # 第100主成分
img = np.reshape(e100,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 3, 3)
e200=eigvec[:,199] # 第200主成分
img = np.reshape(e200,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)
'''


# ここも数字を表示
'''
plt.subplot(1, 5, 4)
X23 = x[:,22,4].T  # 数字5の23番の例．
img = np.reshape(X23,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 5)
s = np.zeros(256)
for i in range(200):
    a = X23.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
    s = s + a*eigvec[:,i]
img = np.reshape(s,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)
'''

# 問題3のコード
'''
for samplenum in range(0,25,5):
    #もとの画像を表示
    plt.subplot(5,5,1+samplenum)
    Xsample = x[:,np.random.randint(500),np.random.randint(10)].T
    img = np.reshape(Xsample,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)

    # 第10主成分で再構成
    plt.subplot(5,5,2+samplenum)
    s = np.zeros(256)
    for i in range(10):
        a = Xsample.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
        s = s + a*eigvec[:,i]
    img = np.reshape(s,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)

    # 第50主成分で再構成
    plt.subplot(5,5,3+samplenum)
    s = np.zeros(256)
    for i in range(50):
        a = Xsample.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
        s = s + a*eigvec[:,i]
    img = np.reshape(s,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)

    # 第100主成分で再構成
    plt.subplot(5,5,4+samplenum)
    s = np.zeros(256)
    for i in range(100):
        a = Xsample.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
        s = s + a*eigvec[:,i]
    img = np.reshape(s,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)

    # 第200主成分で再構成
    plt.subplot(5,5,5+samplenum)
    s = np.zeros(256)
    for i in range(200):
        a = Xsample.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
        s = s + a*eigvec[:,i]
    img = np.reshape(s,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)
'''

# 問題4のコード
r = [0] * eigvec.shape[1] # 基底ベクトル数の要素数を持つリストを作成
X23 = x[:,22,4].T  # 数字5の23番の例．
s = np.zeros(256)
for i in range(eigvec.shape[1]): # 基底ベクトル数の回数分
    a = X23.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
    s = s + a*eigvec[:,i]
    r[i] = np.linalg.norm(X23 - s) # ベクトルのノルムをとる
    

# プロット
plt.title("The deviation between original and reconstruction")
plt.plot(range(eigvec.shape[1]),r,color="red")
plt.xlabel("m : Base image number")
plt.ylabel("r : Deviation")

# 描画した内容を画面表示
plt.show()

