import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# load images
origImg = plt.imread('./data/pcbCropped.png').astype(np.float32)
defectImg = plt.imread('./data/pcbCroppedTranslatedDefected.png').astype(np.float32)

# perform shift
row, col = origImg.shape
rowShift = 10
colShift = 10
registImg = np.zeros(origImg.shape)
registImg[rowShift:,colShift::] = defectImg[0:row-rowShift,0:col-colShift]

# show difference images
diffImg1 = np.abs(origImg-defectImg)
diffImg2 = np.abs(origImg-registImg)
bwImg = diffImg2 > 0.15
height, width = bwImg.shape
border = np.round(0.05*width).astype(np.int32)
borderMask = np.zeros(bwImg.shape).astype(np.float32)
borderMask[border-1:height-border,border-1:width-border] = 1.;
bwImg = bwImg * borderMask


ax = plt.figure()
ax.add_subplot(3,2,1)
plt.imshow(defectImg,cmap='gray')
plt.title('defectImg')
plt.axis('off')
ax.add_subplot(3,2,2)
plt.imshow(registImg,cmap='gray')
plt.title('registImg')
plt.axis('off')
ax.add_subplot(3,2,3)
plt.imshow(diffImg1,cmap='gray')
plt.title('Unaligned diff:origImg-defectImg')
plt.axis('off')
ax.add_subplot(3,2,4)
plt.imshow(diffImg2,cmap='gray')
plt.title('Aligned diff:origImg-registImg')
plt.axis('off')
ax.add_subplot(3,2,5)
plt.imshow(bwImg,cmap='gray')
plt.title('Thresholded + Aligned Difference Image')
plt.axis('off')
plt.show()
plt.imsave('./data/bwImg.png',bwImg,cmap='gray')
