import cv2
import numpy as np

img = cv2.imread('sof.jpg')
img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,127,255,0)
contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>5000:  # remove small areas like noise etc
        hull = cv2.convexHull(cnt)    # find the convex hull of contour
        hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
        if len(hull)==4:
            cv2.drawContours(img,[hull],0,(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()