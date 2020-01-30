# -*- coding:utf-8 -*-
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

# 画像読み込み
originalImg = Image.open('IMG_1904.JPG')
width, height = originalImg.size
# originalImg = originalImg.convert('L') # pillowのグレースケール変換

# ピクセルごとに値を読みだす
img_pixels_rgb = np.array([[originalImg.getpixel((x,y)) for x in range(width)] for y in range(height)])
# グレースケール化後に値を入れる配列を用意
img_pixels = np.array([[ 0 for x in range(width)] for y in range(height)])
# print(img_pixels.shape)
img = Image.new('L', (width, height))
for y in range(height):
    for x in range(width):
        grayscale = img_pixels_rgb[y][x][0] * 0.2126 + img_pixels_rgb[y][x][1] * 0.7152 + img_pixels_rgb[y][x][2] * 0.0722
        img_pixels[y][x] = int(grayscale)
        img.putpixel((x,y),int(grayscale))
img.save("grayscaleImage.png")

FRAGMENT_SIZE = 16
FRAGMENT_NUM = 500
# ランダム抽出 はじめの一回
px = np.random.randint(0,width - FRAGMENT_SIZE)
py = np.random.randint(0,height - FRAGMENT_SIZE)
sample_pixels = np.array([[img_pixels[y + py][x + px] for x in range(FRAGMENT_SIZE)] for y in range(FRAGMENT_SIZE)])

'''
img = Image.new('L', (FRAGMENT_SIZE,FRAGMENT_SIZE))
for y in range(FRAGMENT_SIZE):
    for x in range(FRAGMENT_SIZE):
        img.putpixel((x,y), int(sample_pixels[y][x]))
img.show()
'''

sample_pixels = sample_pixels.reshape(FRAGMENT_SIZE*FRAGMENT_SIZE)
# ランダム抽出
for i in range(1,FRAGMENT_NUM):
    px = np.random.randint(0,width - FRAGMENT_SIZE)
    py = np.random.randint(0,height - FRAGMENT_SIZE)
    sample_pixel = np.array([[img_pixels[y + py][x + px] for x in range(FRAGMENT_SIZE)] for y in range(FRAGMENT_SIZE)])
    sample_pixel = sample_pixel.reshape(FRAGMENT_SIZE*FRAGMENT_SIZE)
    sample_pixels = np.append(sample_pixels,sample_pixel,axis=0)

sample_pixels = sample_pixels.reshape(FRAGMENT_NUM,FRAGMENT_SIZE*FRAGMENT_SIZE)
print(sample_pixels.shape)

z = sample_pixels.T.copy()
print(z.shape)
# 分散・共分散行列 V の計算
V = np.cov(z)
print(V.shape) # (256, 256)

# 正定値対称行列 V の固有ベクトル・固有値の計算
[eigval, eigvec] = np.linalg.eig(V)
print(eigvec.shape) # (256, 256)
print(eigval.shape) # (256,)

# ここで固有ベクトルを固有値の大きい順に並べ替える．
index = np.argsort(eigval)[::-1]
eigvec = eigvec[:,index]
print(eigvec.shape) # (256, 256)
e=eigvec[:,0:2] #最初の2つだけとってきてる
print(e.shape) # (256, 2)

# ここからplot
fig = plt.figure()

'''
for i in range(15):
    plt.subplot(3, 5, i+1)
    e = eigvec[:,i]
    img = np.reshape(e,(16,16))
    plt.imshow(img, cmap=plt.cm.gray_r)

plt.show()

plt.subplot(1, 5, 1)
e1=eigvec[:,0] # 第1主成分
img = np.reshape(e1,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 2)
e2=eigvec[:,1] # 第2主成分
img = np.reshape(e2,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 3)
e50=eigvec[:,49] # 第50主成分
img = np.reshape(e50,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 4)
e100=eigvec[:,99] # 第100主成分
img = np.reshape(e100,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 5)
e200=eigvec[:,199] # 第200主成分
img = np.reshape(e200,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)
'''



# plt.hist(eigval, bins=100)

print(eigval.mean())
print(np.median(eigval))
print(np.percentile(eigval,75))
print(eigval.max())
print(eigval.min())
#print(eigval)
eigval_cp = eigval.copy()



#plt.plot(eigval)


#plt.show()