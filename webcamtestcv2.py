import cv2

rtspURL = "rtsp://172.17.152.80:8554/mystream?rtsp_transport=tcp"
cap = cv2.VideoCapture(rtspURL, cv2.CAP_FFMPEG)

# Increasing ffmpeg probe size and analyze duration
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)


if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame, retrying...")
        break

    print("Frame received:", frame.shape)  # Show frame size instead of GUI

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()