import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

image = cv2.imread("test6.jpg")
original = image.copy()
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    x,y,w,h = cv2.boundingRect(c)
    break

close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
cv2.namedWindow("orginal", cv2.WINDOW_NORMAL)
cv2.imshow("orginal", image) 
cv2.rectangle(close, (x,y), (x+w, y+h), (0, 0, 255), 2)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", close)
cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
cv2.namedWindow("orginalwithbox", cv2.WINDOW_NORMAL)
cv2.imshow("orginalwithbox", image)
cv2.waitKey(0)
