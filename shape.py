import numpy as np
import cv2
import os
import pytesseract
import time

img = cv2.imread("answer4.jpeg")
imGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imGrey, 127, 255, cv2.THRESH_BINARY)

contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
i=1
cImages = []



for contour in contours :
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    

    if len(approx)==4 and x>0 and y>0: #to ignore bigger box

        x,y,w,h = cv2.boundingRect(approx)

        

        if abs(w-h)>10:
            print(x,y,w,h)

            cv2.drawContours(img, [approx], 0, (0,255,0),1)  
            newimg = img[y:y+h , x:x+w]
            cImages.append(newimg)
            cv2.imwrite("Detected Data/Detected Answers/"+str(i)+".jpg", newimg)
            i=i+1

cv2.imshow('BindingBox',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



i=1

for imm in cImages:

    im = cv2.cvtColor(imm,cv2.COLOR_BGR2GRAY )

    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imwrite("Detected Data/Detected Chars/Using OpenCV/"+str(i)+".jpg", im)
    i=i+1



path = "Detected Data/Detected Answers"

# for paths in os.listdir(path):

#     inp_path = os.path.join(path,paths)

#     im = cv2.imread(inp_path,0)

#     ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         x,y,w,z = cv2.boundingRect(cnt)
#         cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
#     cv2.imwrite("Detected Data/Detected Chars/Using OpenCV/"+str(i)+".jpg", im)
#     i=i+1

i=1


for paths in os.listdir(path):

    inp_path = os.path.join(path,paths)

    imm = cv2.imread(inp_path)
    im = cv2.cvtColor(imm, cv2.COLOR_BGR2GRAY)

    height,width = im.shape
    boxes = pytesseract.image_to_boxes(im)

    for b in boxes.splitlines():
        b=b.split(' ')
        x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(im, (x,height-y), (w,height-h), (0,0,0),1)

    cv2.imwrite("Detected Data/Detected Chars/Using Pytesseract/"+str(i)+".jpg", im)
    i=i+1





# cv2.imshow("shapes",cImages[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()