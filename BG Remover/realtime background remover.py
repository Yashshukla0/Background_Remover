import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
segmentor=SelfiSegmentation()
fpsreader= cvzone.FPS()
#imglist=cv2.imread('images/3.jpg')
listImg=os.listdir("images")
print(listImg)
imglist=[]
for i in listImg:
    img=cv2.imread(f'images/{i}')
    imglist.append(img)
indeximg=0

while True:
    s,imaged=cap.read()
    imgout=segmentor.removeBG(imaged,imglist[indeximg],threshold=0.8)
    i=cvzone.stackImages([imaged,imgout],2,1)
    _,i=fpsreader.update(i)

    cv2.imshow('image',i)
    
    key=cv2.waitKey(1)
    if key==ord('a'):
        if indeximg>0:
           indeximg-=1
    elif key==ord('d'):
        if indeximg<len(imglist)-1:
           indeximg+=1
    elif key==ord('q'):
        break
