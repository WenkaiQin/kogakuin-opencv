import cv2

fast = cv2.FastFeatureDetector_create()
print help(fast.detectAndCompute)