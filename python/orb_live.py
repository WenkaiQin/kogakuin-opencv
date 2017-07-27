import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

MIN_MATCH_COUNT = 10

# Initialize ORB detector.
orb = cv2.ORB_create()

# Initialize FLANN parameters.
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) # 6 means FLANN_INDEX_LSH
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Initialize camera.
cap = cv2.VideoCapture(0)

# Initialize target image.
target = cv2.imread('../images/tomare_target.jpg')
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
# target = cv2.pyrDown(target)
target = cv2.flip(target, 1)

# Find keypoints and descriptors of target.
kp1, des1 = orb.detectAndCompute(target, None)

while True:
	# Read and initialize frame.
	im = cap.read()[1]
	im = cv2.pyrDown(im)
	im = cv2.flip(im, 1)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# find the keypoints and descriptors with SIFT
	kp2, des2 = orb.detectAndCompute(im, None)
	matches = flann.knnMatch(des1, des2, k=2)

	good = []

	if len(matches) > 0:
		for match in matches:
			if len(match)==2 and match[0].distance < 0.7*match[1].distance:
					good.append(match[0])

	match_count = '%d/%d' %(len(good), MIN_MATCH_COUNT)
	cv2.putText(im, 'Number of matches: '+match_count, (30,30), 1, 1.5, (70,70,220), 2)

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