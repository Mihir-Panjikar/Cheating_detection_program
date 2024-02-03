import cv2

# if capturing video from phone using 'IP Webcam' APP
# url = 'https://Phone IP address:PORT/video'

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cam.read()

    if ret:
        cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
