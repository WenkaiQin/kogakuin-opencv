import cv2
import numpy as numpy

face_cascade = cv2.CascadeClassifier('haar_classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_classifiers/haarcascade_eye.xml')


cap = cv2.VideoCapture(0)

while True:
	im = cap.read()[1]
	im = cv2.pyrDown(im)
	im = cv2.flip(im, 1)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = im[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

	cv2.imshow('Frame', im)

	if(cv2.waitKey(10)==27):
		print 'Exiting...'
		cv2.destroyAllWindows()
		cap.release()
		break