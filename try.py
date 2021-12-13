import os 
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


imageSamplePath = "./set1/6.jpeg"
imageSample = mpimg.imread(imageSamplePath)

# plt.figure(figsize = (10,10))
# plt.imshow(imageSample , cmap="gray")
# plt.show()



def crop_and_split(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        assert img is not None
    
    # img = img[:, :500]
    img = img[300:,:]
    
    prt = img[:900,200:]
    # hand = img[500:2500]
    return img, prt


imageSample = cv2.imread(imageSamplePath)

x,y = crop_and_split(imageSample)

# cv2.imshow('',x)
# cv2.waitKey(0)

cv2.imshow('',y)
cv2.waitKey(0)