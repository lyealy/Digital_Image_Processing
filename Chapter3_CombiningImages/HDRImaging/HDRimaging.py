import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# load images
def readImages():
    filenames = [
        './data/a2.jpg',
        './data/b2.jpg',
        './data/c2.jpg',
        './data/d2.jpg'
    ]
    images = []
    for file in filenames:
        img = plt.imread(file)
        images.append(img)
    return images

ax = plt.figure()
images = readImages()
for i,image in enumerate(images):
    ax.add_subplot(3,2,i+1)
    plt.imshow(image)
    plt.axis('off')
    
# Align images by median threshold bitmaps (MTB)
alignMTB = cv.createAlignMTB()
alignMTB.process(src=images,dst=images)

# Merge Images
mergeMertens = cv.createMergeMertens()
exposureFusion = mergeMertens.process(images)
ax.add_subplot(3,2,5)
plt.imshow(exposureFusion)
plt.axis('off')
plt.show()

#plt.imsave('./data/exposureFusion.jpg',exposureFusion) #there is some problem when saving image with jpg format
plt.imsave('./data/exposureFusion.png',exposureFusion)
exposureFusion_GBR = cv.cvtColor(exposureFusion,cv.COLOR_RGB2BGR)
cv.imwrite('./data/exposureFusion.jpg',(exposureFusion_GBR*255).astype(np.int32))
#ax.savefig('./data/compare.png')