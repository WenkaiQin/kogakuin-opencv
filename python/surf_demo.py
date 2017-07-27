import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../images/phone_target.jpg') 
# cv2.imshow('Frame', img)

# Initiate FAST object with default values
surf = cv2.xfeatures2d.SURF_create()

# find and draw the keypoints
kp = surf.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
# print "Threshold: ", surf.getThreshold()
# print "Neighborhood: ", surf.getType()
# print "Total Keypoints with nonmaxSuppression: ", len(kp)

# Disable nonmaxSuppression
# surf.setNonmaxSuppression(0)
# kp = surf.detect(img,None)

# print "Total Keypoints without nonmaxSuppression: ", len(kp)
# img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

frame = np.concatenate((img,img2), axis=1)
cv2.imshow('Frame', frame)

cv2.waitKey()
print 'Exiting...'
cv2.destroyAllWindows()