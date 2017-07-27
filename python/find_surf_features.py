import cv2
import matplotlib.pyplot as plt

im = cv2.imread('../images/tomare_target.jpg')
surf = cv2.xfeatures2d.SURF_create(50)
kp, des = surf.detectAndCompute(im, None)
print len(kp)

# im2 = cv2.drawKeypoints(im, kp, None, (255,0,0), 4)
im2 = cv2.drawKeypoints(im, kp, None, color=(255,0,0))
plt.imshow(im2), plt.show()