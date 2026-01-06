import cv2
import numpy as np

# Select the camera (adjust index if needed)
cap = cv2.VideoCapture(1)

# Set resolution
cap.set(3, 1920)
cap.set(4, 1080)

# Create windows
cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Processed (CLAHE + Contrast/Brightness)", cv2.WINDOW_AUTOSIZE)

# Function to do nothing (required for trackbars)
def nothing(x):
    pass

# Create trackbars for live adjustment
cv2.createTrackbar("Clip Limit (x10)", "Processed (CLAHE + Contrast/Brightness)", 20, 50, nothing)  # 0-50 -> 0.0-5.0
cv2.createTrackbar("Tile Size", "Processed (CLAHE + Contrast/Brightness)", 16, 32, nothing)  # 4-32
cv2.createTrackbar("Alpha (x10)", "Processed (CLAHE + Contrast/Brightness)", 15, 30, nothing)  # 5-30 -> 0.5-3.0
cv2.createTrackbar("Beta (-50 to 20)", "Processed (CLAHE + Contrast/Brightness)", 70, 70, nothing)  # 0-70 -> -50 to 20

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get trackbar values
    clip_limit = cv2.getTrackbarPos("Clip Limit (x10)", "Processed (CLAHE + Contrast/Brightness)") / 10.0
    tile_size = cv2.getTrackbarPos("Tile Size", "Processed (CLAHE + Contrast/Brightness)")
    if tile_size < 4:
        tile_size = 4  # Minimum
    alpha = cv2.getTrackbarPos("Alpha (x10)", "Processed (CLAHE + Contrast/Brightness)") / 10.0
    beta = cv2.getTrackbarPos("Beta (-50 to 20)", "Processed (CLAHE + Contrast/Brightness)") - 50

    # Apply CLAHE with live params
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    gray_clahe = clahe.apply(gray)

    # Apply contrast and brightness adjustment
    processed = cv2.convertScaleAbs(gray_clahe, alpha=alpha, beta=beta)

    # Display
    cv2.imshow("Original", frame)
    cv2.imshow("Processed (CLAHE + Contrast/Brightness)", processed)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()