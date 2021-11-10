'''
INSTRUCTIONS

1. Run using python2
2. Create three files names yellow, green and orange in the same directory as as this python file.
2. With the help of left mouse button, first click around yellow buoy 
2. Then press 'Enter (Key which corresponds to 10)'
3. Next, click arond orange buoy and press enter
4. Finally click around green buoy.
5. If orange or green buoy is not present then simply press enter to skip to next
6. Once all three are done you will be able to see the masked images. 
7. Repeat steps for 140 frames.
'''

import cv2 
import numpy as np


global yellowBouyPoints, orangeBouyPoints, greenBouyPoints, currentBouyColour
yellowBouyPoints = []
orangeBouyPoints = []
greenBouyPoints = []
currentBouyColour = 0

# Depending on currentBouyColour, append x,y coordinates in respective lists
def click_and_crop(event, x, y, flags, param):
 	global refPt
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [x, y]
		if currentBouyColour==0:
			yellowBouyPoints.append([x,y])
		elif currentBouyColour==1:
			orangeBouyPoints.append([x,y])
		elif currentBouyColour==2:
			greenBouyPoints.append([x,y])

#Taking points from user
startIndex = 0
stopIndex = 200

yellowPixels = []
orangePixels = []
greenPixels = []
i = -1
for imageIndex in range(startIndex,stopIndex):
	i = i+1
	print ('Image Index ',imageIndex)
	name = "frames/trainingData/frame{num}.jpg".format(num = i)
	image = cv2.imread(name)
	key = 0
	currentBouyColour = 0
	yellowBouyPoints = []
	orangeBouyPoints = []
	greenBouyPoints = []
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	refPt = []
	while key!=27:
		cv2.imshow("image", image)
		try:
			cv2.circle(image,(refPt[0],refPt[1]),1,(255,0,0),2)
		except:
			print ('no points selected')
		key = cv2.waitKey(100)
		# print key
		if currentBouyColour==0:
			print ('select yellow bouy points')
		elif currentBouyColour==1:
			print ('select orange bouy points' )
		elif currentBouyColour==2:
			print( 'select green bouy points')
		if key==10:
			currentBouyColour += 1
		if currentBouyColour>2:
			break
	cv2.destroyAllWindows()

	#Masking
	yellowMask = np.zeros((image.shape[0],image.shape[1]))
	if len(yellowBouyPoints)>3:
		yellowBouyPointsArr = np.array(yellowBouyPoints, np.int32)
		cv2.fillPoly(yellowMask, [yellowBouyPointsArr], 255)
	else:
		print('not enough yellow bouy points')

	orangeMask = np.zeros((image.shape[0],image.shape[1]))
	if len(orangeBouyPoints)>3:
		orangeBouyVerticeArr = np.array(orangeBouyPoints, np.int32)
		cv2.fillPoly(orangeMask, [orangeBouyVerticeArr], 255)
	else:
		print ('not enough orange bouy points')

	greenMask = np.zeros((image.shape[0],image.shape[1]))
	if len(greenBouyPoints)>3:
		greenBouyPointsArr = np.array(greenBouyPoints, np.int32)
		cv2.fillPoly(greenMask, [greenBouyPointsArr], 255)
	else:
		print ('not enough green bouy points')
  	
  	#Show the different masked regions which has been created
	cv2.imshow("yellowMask",yellowMask)
	cv2.imshow("orangeMask",orangeMask)
	cv2.imshow("greenMask",greenMask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Writing the different masked images as png images
	yname = "yellow/y{num}.png".format(num = i)
	oname = "orange/o{num}.png".format(num = i)
	gname = "green/g{num}.png".format(num = i)
	cv2.imwrite(yname, yellowMask)
	cv2.imwrite(oname, orangeMask)
	cv2.imwrite(gname, greenMask)
	
	print (len(yellowPixels))
	print (len(orangePixels))
	print (len(greenPixels))


