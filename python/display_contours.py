import cv2

def main():

	camera = cv2.VideoCapture(0)

	while True:
		(grabbed, frame) = camera.read()

		if not grabbed:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7,7), 0)
		edged = cv2.Canny(blurred, 50, 150)

		# (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cv2.imshow('Edges', blurred)

		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()