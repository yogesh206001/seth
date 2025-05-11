import cv2
import time   
from playsound import playsound

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for detecting drowsiness
frame_check = 30  # Number of consecutive frames eyes should be closed to trigger alert
closed_eye_count = 0  # Counter for closed-eye frames
drowsy_start_time = None
alarm_playing = False

# Function to play alarm sound
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        playsound('alram.mp3')  # Ensure 'alarm.mp3' exists in your working directory
        alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        if len(eyes) == 0:  # If no eyes detected
            closed_eye_count += 1
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
        else:
            closed_eye_count = 0
            drowsy_start_time = None

        # If eyes are closed for a prolonged time, alert and play alarm
        if closed_eye_count > frame_check:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if drowsy_start_time and time.time() - drowsy_start_time > 2:
                print("Drowsiness detected!")
                play_alarm()

    # Display the resulting frame
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

