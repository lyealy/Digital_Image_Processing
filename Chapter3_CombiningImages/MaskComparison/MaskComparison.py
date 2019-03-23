import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

mask1 = plt.imread('./data/mask1.png').astype(np.float32)
mask2 = plt.imread('./data/mask2.png').astype(np.float32)

diffImg = abs(mask1 - mask2).astype(np.int8)
plt.imshow(diffImg,cmap='gray')
plt.show()