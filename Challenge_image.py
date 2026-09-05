import numpy as np
import cv2 

img = cv2.imread('assets/PennAir 2024 App Static.png', -1)
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#takes the image if the pixel value is equal or below this value it will replace it with black
_, thresh = cv2.threshold(img, 220, 250, cv2.THRESH_BINARY)
#detecting the red shape seperately becuase its too dark for thresh
red = cv2.inRange(img, (0, 0, 100), (80, 80, 255))
#I had issues with merging the 1 channel red detection with the 3 channel thresh detection 
#so I'm just gonna copy over the detected red section from the color image
combined = thresh.copy()
combined[red > 0] = img[red > 0]

#need single channel for Contours
threshsingle = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
single = cv2.bitwise_or(threshsingle, red)
contours, hierarchy = cv2.findContours(single, 1, 2) 
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
for cnt in contours:
	#This is what I tried initially but it dosen't work for shapes like triangle 
	#where the center is not the same as the center of the box around the shape
	#x, y, w, h = cv2.boundingRect(cnt)
	#midx = int(x+1/2*w)
	#midy = int(y + 1/2*h)
	
	midx = int(cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'])
	midy = int(cv2.moments(cnt)['m01'] / cv2.moments(cnt)['m00'])
	cv2.circle(img, (midx, midy), 5, (0, 0, 0), -1)
	cv2.putText(img, "center", (midx - 15, midy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
	
cv2.imshow('Penn Air analyzed image', img)  
cv2.waitKey(0) 
cv2.destroyAllWindows()

#abandoned approach
#shi Tomaski corner detection algorithm
#corners = cv2.goodFeaturesToTrack(img, 100, 0.1, 10)
#corners = np.intp(corners)

#for corner in corners:
	#x, y = corner.ravel() 
	#cv2.circle(img, (x, y), 5,  (255, 0, 0), -1)  
