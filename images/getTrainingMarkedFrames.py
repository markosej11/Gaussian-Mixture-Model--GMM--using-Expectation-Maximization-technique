'''
------------
INSTRUCTIONS
------------
GUI to extract pixel values for each bouy
Instruction:
1. Click edges pixels around yellow bouy first
2. Press 'Enter'
3. If orange bouy is present, click edges pixels around the orange bouy, else press 'Enter'. 
   If orange bouy is present, choose the edge pixels and then press 'Enter'
4. If green bouy is present, click edges pixels around the green bouy, else press 'Enter'. 
   If green bouy is present, choose the edge pixels and then press 'Enter'
'''

import cv2 
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle

global yellowBouyVertices, orangeBouyVertices, greenBouyVertices, currBouyColor
yellowBouyVertices = []
orangeBouyVertices = []
greenBouyVertices = []
currBouyColor = 0

# def lineFromPoints(points):
# 	# Example for mat for passing points argument, points = [(163,52),(170,90)]
# 	x_coords, y_coords = zip(*points)
# 	A = vstack([x_coords,ones(len(x_coords))]).T
# 	m, c = lstsq(A, y_coords)[0]
# 	# print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
# 	return m,c

def click_and_crop(event, x, y, flags, param):
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	global refPt
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [x, y]
		if currBouyColor==0:
			yellowBouyVertices.append([x,y])
		elif currBouyColor==1:
			orangeBouyVertices.append([x,y])
		elif currBouyColor==2:
			greenBouyVertices.append([x,y])

trainingImagesNames = glob.glob('frames/trainingData/*.jpg')
trainingImagesNames.sort()
print (len(trainingImagesNames))


startIndex = 0
stopIndex = 40

yellowPixels = []
orangePixels = []
greenPixels = []
# print range(startIndex,stopIndex)
i = -1
for imageIndex in range(startIndex,stopIndex):
	i = i+1
	print ('Image Index ',imageIndex)
	name = "frames/trainingData/frame{num}.jpg".format(num = i)
	image = cv2.imread(name)

	# image = cv2.imread(trainingImagesNames[imageIndex])
	key = 0
	currBouyColor = 0
	yellowBouyVertices = []
	orangeBouyVertices = []
	greenBouyVertices = []
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	refPt = []
	while key!=27:
		cv2.imshow("image", image)
		try:
			cv2.circle(image,(refPt[0],refPt[1]),1,(255,0,0),2)
		except:
			print ('no points chosen')
		key = cv2.waitKey(100)
		# print key
		if currBouyColor==0:
			print ('select yellow bouy points')
		elif currBouyColor==1:
			print ('select orange bouy points' )
		elif currBouyColor==2:
			print( 'select green bouy points')
		if key==10:
			# 0=yellow 1=orange 2=green
			currBouyColor += 1
		if currBouyColor>2:
			break
	cv2.destroyAllWindows()

	originalImage = cv2.imread(trainingImagesNames[imageIndex])
	yellowMask = np.zeros((image.shape[0],image.shape[1]))
	if len(yellowBouyVertices)>3:
		yellowBouyVerticesArr = np.array(yellowBouyVertices, np.int32)
		cv2.fillPoly(yellowMask, [yellowBouyVerticesArr], 255)

	orangeMask = np.zeros((image.shape[0],image.shape[1]))
	if len(orangeBouyVertices)>3:
		orangeBouyVerticeArr = np.array(orangeBouyVertices, np.int32)
		cv2.fillPoly(orangeMask, [orangeBouyVerticeArr], 255)
	else:
		print ('not enough orange bouy points chosen')

	greenMask = np.zeros((image.shape[0],image.shape[1]))
	if len(greenBouyVertices)>3:
		greenBouyVerticesArr = np.array(greenBouyVertices, np.int32)
		cv2.fillPoly(greenMask, [greenBouyVerticesArr], 255)
	else:
		print ('not enough green bouy points chosen')
  	
	cv2.imshow("yellowMask",yellowMask)
	cv2.imshow("orangeMask",orangeMask)
	cv2.imshow("greenMask",greenMask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	yname = "yellow/y{num}.png".format(num = i)
	oname = "orange/o{num}.png".format(num = i)
	gname = "green/g{num}.png".format(num = i)
	cv2.imwrite(yname, yellowMask)
	cv2.imwrite(oname, orangeMask)
	cv2.imwrite(gname, greenMask)
	

	for xIndex in range(image.shape[1]):
		for yIndex in range(image.shape[0]):
				if yellowMask[yIndex,xIndex]:
					yellowPixels.append(originalImage[yIndex,xIndex])
				if orangeMask[yIndex,xIndex]:
					orangePixels.append(originalImage[yIndex,xIndex])
				if greenMask[yIndex,xIndex]:
					greenPixels.append(originalImage[yIndex,xIndex])

	print (len(yellowPixels))
	print (len(orangePixels))
	print (len(greenPixels))

with open('trainingPixels.pkl', 'wb') as f:
	pickle.dump([yellowPixels,orangePixels,greenPixels], f)
