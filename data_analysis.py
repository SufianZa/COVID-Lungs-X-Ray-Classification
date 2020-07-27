import os
import random
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from collections import Counter


# Read the image list and csv
images_path = 'images/*.jpg'
image_files = glob(images_path, recursive=True)
df = pd.read_csv('metadata.csv')
df = df[~df.isna()]

# print csv features
print(df.columns)

colored_imgs = []
gray_imgs = []
for im in image_files:
  image = np.array(Image.open(im))
  if image.ndim > 2:
    colored_imgs.append((image.shape[0],image.shape[1], image.shape[2]))
  else:
    gray_imgs.append((image.shape[0],image.shape[1], None))


data = np.array(colored_imgs + gray_imgs)
colored_imgs = np.array(colored_imgs)
gray_imgs = np.array(gray_imgs)

print('RGB images {}'.format(len(colored_imgs)))
print('Grayscale images {}'.format(len(gray_imgs)))
fig = plt.figure(figsize=(30, 10))
ax1 = fig.add_subplot(121)

plt.figure(figsize=(30, 10))
ax1.scatter(colored_imgs[:, 0], colored_imgs[:, 1], c='r', label='RGB Images')
ax1.scatter(gray_imgs[:, 0], gray_imgs[:, 1], c='gray', label='Grayscale Images')
ax1.grid()
ax1.set_xlabel('width', fontsize=15)
ax1.set_ylabel('height', fontsize=15)
ax1.set_title('Images Sizes and dimensions (Pixels)', fontsize=15)
ax1.legend()
ax1 = fig.add_subplot(122)

N, edges = np.histogram(data[:, :1], bins=100)
edges = edges[1:] - (edges[2] - edges[1]) / 2;
ax1.plot(edges, N)
ax1.grid()
ax1.set_xlabel('Image size', fontsize=15)
ax1.set_ylabel('Num of Images', fontsize=15)
ax1.xaxis.set_major_formatter(FuncFormatter((lambda x, b: '({},{})'.format(int(x), int(x)))))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title('Images Sizes distripution', fontsize=15)
plt.savefig('image_sizes.png')

plt.figure(figsize=(15,5))
plt.subplot(121)
male_ratio = df[(df["Covid"] == 1) & (df["Sex"] == 'Male')]
female_ratio = df[(df["Covid"] == 1) & (df["Sex"] == 'Female')]
plt.bar('Male', male_ratio.shape[0])
plt.bar('Female', female_ratio.shape[0])
plt.ylabel('Cases', fontsize=15)

plt.subplot(122)
covid = df[(df["Covid"] == 1)]
plt.bar(np.array(covid.groupby('Age')['Age'])[:,0], covid.groupby('Age')['Age'].count())
plt.xlabel('Age', fontsize=15)
plt.ylabel('Cases', fontsize=15)
plt.savefig('features_analysis.png')
