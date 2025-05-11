import cv2
import numpy as np
import datetime

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam; replace with video file path if needed

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Video writer initialization variables
recording = False
out = None

# Function to initialize the video writer
def start_recording(frame):
    global out
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"motion_{timestamp}.avi"
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    print(f"Recording started: {filename}")

while True:
    # Read frames from the video capture object
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Reduce noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust the threshold for detecting motion
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Start recording if motion is detected
    if motion_detected:
        if not recording:
            start_recording(frame)
            recording = True
        if recording:
            out.write(frame)  # Save the frame to the video file
    else:
        if recording:
            print("Motion stopped. Recording ended.")
            recording = False
            out.release()  # Stop recording

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Exit when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()