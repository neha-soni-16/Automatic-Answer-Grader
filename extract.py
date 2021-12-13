
import numpy as np
import pandas as pd
import os 
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ans_path = "./answers"

for i in os.listdir(ans_path) :
    os.remove(os.path.join(ans_path + "/" + i))


#imageSamplePath = "./dataset/A/a1.jpg"
imageSamplePath = "./set1/9.jpeg"
ansImage = mpimg.imread(imageSamplePath)

plt.figure(figsize = (10,10))
plt.imshow(ansImage , cmap="gray")
plt.show()

print(ansImage.shape)

image = cv2.imread(imageSamplePath)

imageSample = image


# def crop_and_split(img):
#     if isinstance(img, str):
#         img = cv2.imread(img)
#         assert img is not None
    
#     # img = img[:, :500]
#     img = img[300:,:]
    
#     prt = img[:900,200:]
#     # hand = img[500:2500]
#     return img, prt

# _, imageSample = crop_and_split(image)

from scipy.spatial.distance import cdist

def order_points(pts):
    x = pts[np.argsort(pts[:, 0]), :]

    left = x[:2, :]
    right = x[2:, :]

    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left

    d = cdist(tl[np.newaxis], right, "euclidean")[0]
    (br, tr) = right[np.argsort(d)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def find_chars(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, th = cv2.threshold(img, 0, 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c[:, 0, :] for c in contours]
    return contours

def crop_min_area_rect(rect, img):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Reorder points to [tl, tr, br, bl]
    box = order_points(box).astype("float32")
    w = int(np.linalg.norm(box[0] - box[1]))
    h = int(np.linalg.norm(box[0] - box[3]))
    
    dst_pts = np.array([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def extract_lines(img, warp_crop=False, padx=4, pady=4):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Finding text lines
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255 , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    block_kern_shape = (100, 1)
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, block_kern_shape)
    dilation = cv2.dilate(th, block_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    crops, chars, bboxes = [], [], []
    for ctr in contours:
        box = cv2.boundingRect(ctr)
        if warp_crop:
            rect = cv2.minAreaRect(ctr)
            crop = crop_min_area_rect(rect, img)
            if is_unwanted(crop.shape[0], crop.shape[1]):
                continue
        else:
            x, y, w, h = box
            if is_unwanted(h, w):
                continue
            crop = img[y-pady:y+h+pady, x-padx:x+w+padx]

        char = find_chars(crop)
        if len(char) < 3:
            continue

        crops.append(crop)
        chars.append(char)
        bboxes.append(box)

    # crops, chars, bboxes = filter_crops(crops, chars, bboxes)
    return crops, chars, bboxes

def is_unwanted(h, w):
    return any([h < 10, w < 200, w/h < 1, w/h > 100])

img = imageSample.copy()
crops, contours, bboxes = extract_lines(img)

k = int(np.ceil(len(crops) / 4))
_, axs = plt.subplots(k, 4, figsize=(25, 5))
for crop, box, ax in zip(crops, bboxes, axs.flatten()):
    ax.imshow(crop, cmap='gray', vmin=0, vmax=255)
    ax.set_title(box)
    
plt.show()

_, axs = plt.subplots(k, 4, figsize=(25, 5))
for crop, ctrs, ax in zip(crops, contours, axs.flatten()):
    _img = cv2.drawContours(np.zeros(crop.shape[:2], dtype='uint8'), ctrs, -1, (255), -1)
    ax.imshow(_img, cmap='gray', vmin=0, vmax=255)
    
plt.show()

_, axs = plt.subplots(k, 4, figsize=(25, 5))
for box, ax in zip(bboxes, axs.flatten()):
    x, y, w, h = box
    ax.imshow(img[y:y+h, x:x+w])
    
plt.show()

_img = img.copy()
for (x, y, w, h) in bboxes:
    _img = cv2.rectangle(_img, (x, y), (x+w, y+h), 2)
plt.imshow(_img)

plt.show()

_img = img.copy()
for (x, y, w, h) in bboxes:
    _img = cv2.rectangle(_img, (x, y), (x+w, y+h), 2)


plt.figure(figsize = (10,10))
plt.imshow(_img)
plt.show()

i = 0
for image in crops:
    cv2.imwrite("./answers/" + str(i) + ".jpeg", image)
    i = i + 1


answer_dir ="./answers"

data = []

file = open("result.txt","r+")
file.truncate(0)
file.close()

for a in os.listdir(answer_dir) : 
    a_path = os.path.join(answer_dir + "/" + a)

    out = os.system("python3 main.py --img_file %s >> result.txt" % a_path)
    

print("\n \nOutput data \n\n")
out = open("result.txt")
print(out.read())


os.system("g++ -o grade grade.cpp | ./grade")


