import cv2
import numpy as np

# Load the cascade files
face_cascade = cv2.CascadeClassifier('D:\SYMPOSIUM COURSE\proj\ML\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:\SYMPOSIUM COURSE\proj\ML\haarcascades\haarcascade_eye.xml')

# Check if the cascade files have been loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    raise IOError('Unable to load cascade classifier XML files')

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Scaling factors
scaling_factor = 0.5


while True:
    # Read frames from the camera
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale frame
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces and eyes
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Extract the grayscale face ROI
        roi_gray = gray[y:y + h, x:x + w]
        # Extract the color face ROI
        roi_color = frame[y:y + h, x:x + w]

        # Run the eye detector on the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Draw circles around the detected eyes
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

    # Display the combined output
    cv2.imshow('Face and Eye Detector', frame)

    # Check if the user pressed the 'Esc' key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
