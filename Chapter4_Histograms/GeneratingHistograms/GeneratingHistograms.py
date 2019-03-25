import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

imgbay = plt.imread('./data/bay.jpg')
imgbrain = plt.imread('./data/brain.jpg')
# openCV function is much faster than the corresponding numpy function
# hist = cv.calcHist(images = [img],channels = [0],mask = None,histSize=[256],ranges=[0,256])
# print(hist.shape)
fig = plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(imgbay,cmap='gray')
plt.axis('off')
fig.add_subplot(2,2,2)
plt.hist(imgbay.reshape(-1,),256,[0,256])
fig.add_subplot(2,2,3)
plt.imshow(imgbrain,cmap='gray')
plt.axis('off')
fig.add_subplot(2,2,4)
plt.hist(imgbrain.reshape(-1,),256,[0,256])
plt.show()
