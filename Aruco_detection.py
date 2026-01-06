# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities
import time
# This is the Aruco library from OpenCV
import cv2.aruco as aruco 
import logging
from collections import deque
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# Function to compute circular mean of angles (avoid error when angles flip between -179 and +180 degrees)
def circular_mean(angles_rad):
    """
    Circular mean of angles (radians).
    Returns mean angle in [-pi, pi].

    angles_rad: iterable of floats (radians)
    """
    angles = np.asarray(list(angles_rad), dtype=float)
    if angles.size == 0:
        return 0.0
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return float(np.arctan2(s, c))


Camera=np.load('Calibration.npz') #Load the camera calibration values 
CM=Camera['CM'] #camera matrix 
dist_coef=Camera['dist_coef']# distortion coefficients from the camera 

# Define the ArUco marker parameters
ROBOT_ID = 0                 # <-- change this to your robot marker ID
TARGET_IDS = {1,2,3,4,5}     # <-- change these to your five target IDs
MARKER_LENGTH_M = 77.5      # marker size in mm

# Buffers for averaging distance and bearing over last 20 frames
buffers = {tid: {'dist': deque(maxlen=20), 'bear': deque(maxlen=20)} for tid in TARGET_IDS}


# Load the ArUco Dictionary Dictionary 4x4_50 and set the detection parameters 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
pa = aruco.DetectorParameters() 

# Select the first camera (0) that is connected to the machine
# in Laptops should be the build-in camera
cap = cv2.VideoCapture(1)
 
# Set the width and heigth of the camera to 1920x1080
cap.set(3,500)
cap.set(4,500)

#Create three opencv named windows
cv2.namedWindow("aruco-image", cv2.WINDOW_AUTOSIZE)

#Position the windows next to eachother
cv2.moveWindow("aruco-image",640,100)

# Function to wrap angles to [-pi, pi] ensuring they are comparable
def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2*np.pi) - np.pi

# Function to extract yaw from rotation vector
def yaw_from_rvec(rvec: np.ndarray) -> float:
    """
    Extract yaw (rotation about camera Z axis) from rvec.
    For a top-down camera, this corresponds well to heading on the floor plane.
    Returns radians.
    """
    R, _ = cv2.Rodrigues(rvec)
    # yaw for ZYX convention
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw

def marker_pixel_centre(corners_4x2: np.ndarray) -> tuple[float, float]:
    # corners come as (4,2)
    c = corners_4x2.mean(axis=0)
    return float(c[0]), float(c[1])


# Execute this continuously
while(True):
    
    # Start the performance clock
    start = time.perf_counter()
    
    # Capture current frame from the camera
    ret, frame = cap.read()

    # Convert the image from the camera to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Further adjust contrast and brightness to enhance black-white separation
    gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-50)

    # Apply CLAHE to increase contrast for better ArUco detection
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(32, 32))
    gray = clahe.apply(gray)



    # Detect the markers in the gray image    
    corners, ids, rP = aruco.detectMarkers(gray, aruco_dict) 

    # Calculate the pose of the marker based on the Camera calibration
    rvecs,tvecs,_objPoints = aruco.estimatePoseSingleMarkers(corners,MARKER_LENGTH_M,CM,dist_coef) 

    # Draw the detected markers as an overlay on the original frame    
    out = aruco.drawDetectedMarkers(gray, corners, ids)     

    # Initialize dictionaries to hold marker positions and orientations
    marker_xy = {}
    marker_yaw = {}
    marker_px = {}

    # If there are markers detected, draw the axis for each marker
    if ids is not None:
        # draw axes for each detected marker and log its id and translation vector (tvec)
        # rvecs/tvecs have shape (N,1,3) so we extract the inner 3-vector for each marker
        for i, marker_id in enumerate(ids.flatten()):
            try:
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]    
                # Store 2D position on the assumed plane (camera frame X-Y)
                x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                marker_xy[int(marker_id)] = (x, y)

                # Store yaw (heading)
                marker_yaw[int(marker_id)] = float(yaw_from_rvec(rvec))
    
            except Exception:
                # defensive: skip if shapes are unexpected
                continue
            cv2.drawFrameAxes(out, CM, dist_coef, rvec, tvec, 10)

            # Pixel centre for debug drawing
            c = corners[i][0]  # (4,2)
            marker_px[marker_id] = marker_pixel_centre(c)   

            # log marker id and translation vector with limited precision
            logging.info("Detected marker id=%d, tvec=%s", int(marker_id), np.array2string(tvec, precision=3, separator=', '))

        if ROBOT_ID in marker_xy:
            xr, yr = marker_xy[ROBOT_ID]
            theta_r = marker_yaw.get(ROBOT_ID, 0.0)

            for tid in TARGET_IDS:
                if tid not in marker_xy:
                    continue

                xt, yt = marker_xy[tid]
                dx = xt - xr
                dy = yt - yr

                distance = float(np.hypot(dx, dy))                 # metres (if MARKER_LENGTH_M is metres)
                angle_world = float(np.arctan2(dy, dx))            # radians
                bearing = float(wrap_to_pi(angle_world - theta_r)) # radians, relative to robot heading

                # Append to buffers
                buffers[tid]['dist'].append(distance)
                buffers[tid]['bear'].append(bearing)

                # Use averaged values if buffer is full, else current
                if len(buffers[tid]['dist']) == 20:
                    display_distance = sum(buffers[tid]['dist']) / 20
                    display_bearing = circular_mean(buffers[tid]['bear'])  # Simple average; for angles, consider circular mean if variations are large
                else:
                    display_distance = distance
                    display_bearing = bearing

                logging.info(
                    "Robot->Target %d: distance=%.3f mm, bearing=%.2f deg (theta_r=%.2f deg)",
                    tid, display_distance, np.degrees(display_bearing), np.degrees(theta_r)
                )

                # Draw line and text on the output image
                if ROBOT_ID in marker_px and tid in marker_px:
                    xrp, yrp = marker_px[ROBOT_ID]
                    xtp, ytp = marker_px[tid]
                    cv2.line(out, (int(xrp), int(yrp)), (int(xtp), int(ytp)), (255, 255, 255), 2)
                    cv2.putText(
                        out,
                        f"d={display_distance:.2f}m b={np.degrees(display_bearing):.0f}deg",
                        (int(xtp) + 10, int(ytp) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )


    # Apply Canny edge detection to the gray image
    #canny = cv2.Canny(gray,100,200) 
    
    # Display the resulting frames in the created windows

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