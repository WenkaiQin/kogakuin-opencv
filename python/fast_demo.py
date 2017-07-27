import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../images/phone_target.jpg') 
# cv2.imshow('Frame', img)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print "Threshold: ", fast.getThreshold()
print "Neighborhood: ", fast.getType()
print "Total Keypoints with nonmaxSuppression: ", len(kp)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

frame = np.concatenate((img,img2,img3), axis=1)
cv2.imshow('Frame', frame)

cv2.waitKey()
print 'Exiting...'
cv2.destroyAllWindows()