import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

imgbay = plt.imread('./data/bay.jpg')
imgbrain = plt.imread('./data/brain.jpg')
imgmoon = plt.imread('./data/moon.jpg')

# global histgram equalization or Contrast Limited Adaptive Histogram Equalization
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgbay = np.concatenate([imgbay,cv.equalizeHist(imgbay),clahe.apply(imgbay)], axis = 1)
imgbrain = np.concatenate([imgbrain,cv.equalizeHist(imgbrain),clahe.apply(imgbrain)], axis = 1)
imgmoon = np.concatenate([imgmoon,cv.equalizeHist(imgmoon),clahe.apply(imgmoon)], axis = 1)

fig = plt.figure()
fig.add_subplot(3,1,1)
plt.imshow(imgbay,cmap='gray')
plt.axis('off')
fig.add_subplot(3,1,2)
plt.imshow(imgbrain,cmap='gray')
plt.axis('off')
fig.add_subplot(3,1,3)
plt.imshow(imgmoon,cmap='gray')
plt.axis('off')
plt.show()