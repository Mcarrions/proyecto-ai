import cv2
import numpy as np

img = cv2.imread('2.jpg')

scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)


corners = cv2.goodFeaturesToTrack(gray, 4, 0.1, 200)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(gray,(x-10,y-10),(x+10,y+10),(0,255,0), 2)
	
	
cv2.imshow("Corners Found", gray)
cv2.waitKey()
cv2.destroyAllWindows()