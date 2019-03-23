import numpy as  np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

mask = plt.imread('./data/mask.jpg').astype(np.float32)
live = plt.imread('./data/live.jpg').astype(np.float32)
diff = np.abs(mask-live).astype(np.uint8)

# enhance contrast
clahe = cv.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
diff_en = clahe.apply(diff)


ax = plt.figure()
ax.add_subplot(1,4,1)
plt.imshow(mask,cmap='gray')
plt.title('mask')
plt.axis('off')
ax.add_subplot(1,4,2)
plt.imshow(live,cmap='gray')
plt.title('live')
plt.axis('off')
ax.add_subplot(1,4,3)
plt.imshow(diff,cmap=cm.gray)
plt.title('diff')
plt.axis('off')
ax.add_subplot(1,4,4)
plt.imshow(diff_en,cmap='gray')
plt.title('diff_en')
plt.axis('off')
plt.show()

