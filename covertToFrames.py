'''
INSTRUCTIONS

1. Make sure the Data set is in the same directory as this python file
'''

import cv2   
# Function to convert video into frames 
def FrameCapture(path): 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1
    while success: 
        # Reading frame by frame 
        success, image = vidObj.read() 
        #Saving frame frame by frame
        cv2.imwrite("frame%d.jpg" % count, image) 
        count += 1
  
# Main
if __name__ == '__main__': 
  
    # Calling function with the video path
    FrameCapture("detectbuoy.avi")