import numpy as np
import cv2 
fx = 2564.3186869
fy = 2569.70273111

cap = cv2.VideoCapture('assets/PennAir 2024 App Dynamic.mp4')

while True:
	ret, img = cap.read()
	if not ret:
		break
		
	#takes the image if the pixel value is equal or below this value it will replace it with black
	_, thresh = cv2.threshold(img, 150, 250, cv2.THRESH_BINARY)
	#detecting the red shape seperately becuase its too dark for thresh
	red = cv2.inRange(img, (0, 0, 100), (80, 80, 255))
	#I had issues with merging the 1 channel red detection with the 3 channel thresh detection 
	#so I'm just gonna copy over the detected red section from the color image
	#combined = thresh.copy()
	#combined[red > 0] = img[red > 0]
	
	#need single channel for Contours
	threshsingle = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	single = cv2.bitwise_or(threshsingle, red)
	kernel = np.ones((5, 5), np.uint8)
	single = cv2.morphologyEx(single, cv2.MORPH_OPEN, kernel, iterations=2)
	contours, hierarchy = cv2.findContours(single, 1, 2)
	contours = [c for c in contours if cv2.contourArea(c) > 100]
	cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	Z = None
	Z = None
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		aspect = w / float(h)
		area = cv2.contourArea(cnt)
		if 0.85 < aspect < 1.15 and area > 5000:
			radius_px = w / 2
			Z = fx * 10 / radius_px
			break
	print("Z:", Z)
	for cnt in contours:
		#This is what I tried initially but it dosen't work for shapes like triangle 
		#where the center is not the same as the center of the box around the shape
		#x, y, w, h = cv2.boundingRect(cnt)
		#midx = int(x+1/2*w)
		#midy = int(y + 1/2*h)
		#running into divide by 0 error:
		if cv2.moments(cnt)['m00'] == 0:
			continue 
		midx = int(cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'])
		midy = int(cv2.moments(cnt)['m01'] / cv2.moments(cnt)['m00'])
		cv2.circle(img, (midx, midy), 5, (0, 0, 0), -1)
		if Z:
			label = f"({midx*Z/fx:.1f}, {midy*Z/fy:.1f}, {Z:.1f})"
		else:
			label = "center"
		cv2.putText(img, label, (midx - 15, midy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
	cv2.imshow('Penn Air analyzed image', img)  
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

