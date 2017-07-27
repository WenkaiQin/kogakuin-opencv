## DOESN'T WORK BECAUSE FAST DOES NOT COMPUTE DESCRIPTORS

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

MIN_MATCH_COUNT = 10

# Initialize SIFT detector.
fast = cv2.FastFeatureDetector_create()

# Initialize FLANN parameters.
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Initialize camera.
cap = cv2.VideoCapture(0)

# Initialize target image.
target = cv2.imread('../images/phone_target.jpg')
target = cv2.pyrDown(target)
target = cv2.flip(target, 1)

# Find keypoints and descriptors of target.
kp1, des1 = fast.detectAndCompute(target, None)

while True:
	# Read and initialize frame.
	im = cap.read()[1]
	im = cv2.pyrDown(im)
	im = cv2.flip(im, 1)

	# find the keypoints and descriptors with SIFT
	kp2, des2 = fast.detectAndCompute(im, None)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []

	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		h,w = target.shape[:2]
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts, M)

		im = cv2.polylines(im, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

	else:
		match_count = '%d/%d' %(len(good), MIN_MATCH_COUNT)
		cv2.putText(im, 'Not enough matches are found: '+match_count, (30,30), 1, 1.5, (70,70,220), 2)
		print "Not enough matches are found - %d/%d" %(len(good), MIN_MATCH_COUNT)
		matchesMask = None

	draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matchesMask, flags=2)
	img3 = cv2.drawMatches(target, kp1, im, kp2, good, None, **draw_params)

	# plt.imshow(img3, 'gray'), plt.show()
	cv2.imshow('Frame', img3)

	k = cv2.waitKey(5)
	if k == 27:
		print 'Exiting...'
		cap.release()
		break