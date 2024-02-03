import cv2
import os

# if capturing video from phone using 'IP Webcam' APP
# PHONE_URL = 'https://Phone IP address:PORT/video'

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(os.environ["PHONE_URL"])

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cam.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = frame[y:y+h, x:x+h]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
                cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
