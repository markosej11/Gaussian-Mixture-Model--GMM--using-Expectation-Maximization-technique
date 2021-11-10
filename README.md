ENPM673: Perception for Autonomous Robotics - Project 3

Instructions to run the program:

    python -u project3.py

Following the execution of the program, the animation of the result will play.
Then the result will be written to an output file.
Only the above line needs to be executed to see the output. 

file name 1: project3.py (To see the final result for the project)
file name 2: covertToFrames.py (To convert video dataset to frames)
file name 3: getTrainingMarkedFrames.py (To create yellow, orange, green masked images from traing data)
file name 4: expMax.py (Expectation maximaization. This code is called by project3.py during runtime)

Additional files running instructions (Not necessary for you to run these files to get the final output) :
covertToFrames.py 
1. Make sure the video dataset and py file are in same directory.
2. Simply run the file to covert video into frames.

getTrainingData.py
1. Run using python2
2. Create three files names yellow, green and orange in the same directory as as this python file.
3. With the help of left mouse button, first click around yellow buoy 
4. Then press 'Enter (Key which corresponds to 10)'
5. Next, click arond orange buoy and press enter
6. Finally click around green buoy.
7. If orange or green buoy is not present then simply press enter to skip to next
8. Once all three are done you will be able to see the masked images. 
9. Repeat steps for 140 frames.


We imported the following libraries:
* numpy
* scipy
* cv2
* sys
* os
* argparse


The video output of our program can be found at the link below:
https://drive.google.com/open?id=1g7n3CDNtIvtxrhJRqnG_eTM6I_I1znce
