# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities
import time
# This is the Aruco library from OpenCV
import cv2.aruco as aruco 
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

Camera=np.load('Sample_Calibration.npz') #Load the camera calibration values 
CM=Camera['CM'] #camera matrix 
dist_coef=Camera['dist_coef']# distortion coefficients from the camera 

# Load the ArUco Dictionary Dictionary 4x4_50 and set the detection parameters 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
pa = aruco.DetectorParameters() 

# Select the first camera (0) that is connected to the machine
# in Laptops should be the build-in camera
cap = cv2.VideoCapture(0)
 
# Set the width and heigth of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)

#Create three opencv named windows
cv2.namedWindow("frame-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("gray-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("canny-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("aruco-image", cv2.WINDOW_AUTOSIZE)

#Position the windows next to eachother
cv2.moveWindow("frame-image",0,100)
cv2.moveWindow("gray-image",640,100)
cv2.moveWindow("canny-image",640,100)
cv2.moveWindow("aruco-image",640,100)

# Execute this continuously
while(True):
    
    # Start the performance clock
    start = time.perf_counter()
    
    # Capture current frame from the camera
    ret, frame = cap.read()

    # Convert the image from the camera to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the gray image    
    corners, ids, rP = aruco.detectMarkers(gray, aruco_dict) 

    # Calculate the pose of the marker based on the Camera calibration
    rvecs,tvecs,_objPoints = aruco.estimatePoseSingleMarkers(corners,70,CM,dist_coef) 

    # Draw the detected markers as an overlay on the original frame    
    out = aruco.drawDetectedMarkers(frame, corners, ids)     

    # If there are markers detected, draw the axis for each marker
    if ids is not None:
        # draw axes for each detected marker and log its id and translation vector (tvec)
        # rvecs/tvecs have shape (N,1,3) so we extract the inner 3-vector for each marker
        for i, marker_id in enumerate(ids.flatten()):
            try:
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
            except Exception:
                # defensive: skip if shapes are unexpected
                continue
            cv2.drawFrameAxes(out, CM, dist_coef, rvec, tvec, 10)
            # log marker id and translation vector with limited precision
            logging.info("Detected marker id=%d, tvec=%s", int(marker_id), np.array2string(tvec, precision=3, separator=', '))

    # Apply Canny edge detection to the gray image
    canny = cv2.Canny(gray,100,200) 
    
    # Display the original frame in a window
    cv2.imshow('frame-image',frame)

    # Display the grey image in another window
    cv2.imshow('gray-image',gray)
    
    # Display the canny image in another window
    cv2.imshow('canny-image',canny)

    cv2.imshow('aruco-image',out)

    # Stop the performance counter
    end = time.perf_counter()
    
    # Print to console the execution time in FPS (frames per second)
    fps = 1.0 / (end - start) if end > start else 0.0
    logging.info("FPS: %.1f", fps)

    # If the button q is pressed in one of the windows 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Exit the While loop
        break
    

# When everything done, release the capture
cap.release()
# close all windows
cv2.destroyAllWindows()
# exit the kernel
exit(0)