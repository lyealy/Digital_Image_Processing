# Image Averaging for Noise Reduction
import matplotlib.pyplot as plt
import numpy as np
import gc
ax1 = plt.figure(1,figsize=(100,200))
img = plt.imread('./data/quadnight.JPG')
ax1.add_subplot(1,2,1)
plt.imshow(img)
ax1.add_subplot(1,2,2)
img = (img-127.5)/127.5
plt.imshow(img)



ax2 = plt.figure(2)
num_v = [1,5,15,20] # number of images to average
for i,num in enumerate(num_v):
    imgAvg = np.zeros(img.shape,dtype=np.float32)
    for _ in range(num):
        noiseImg = img + np.random.normal(0,0.1,img.shape)
        imgAvg = imgAvg + noiseImg
    # if image dtype is float32, then imshow only accept value from 0-1
    # if image dtype is int, then imshow only accept value from 0-255
    imgAvg = ((imgAvg/num)*127.5 + 127.5).astype(np.int)
    ax2.add_subplot(2,2,i+1)
    plt.imshow(imgAvg)
    plt.axis('off')
plt.show()
