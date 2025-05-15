'''
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
'''

import cv2
import subprocess
import numpy as np

rtspURL = "rtsp://172.17.152.80:8554/mystream?rtsp_transport=tcp"

# Run FFmpeg as a subprocess to read RTSP stream and output raw frames
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,             # Input RTSP URL
    "-f", "image2pipe",        # Output as raw frames
    "-pix_fmt", "bgr24",       # Ensure OpenCV-compatible format
    "-vcodec", "rawvideo",     # Raw video format
    "-an",                     # No audio,   # Resize if needed
    "pipe:1"                   # Output to pipe
]

# Start FFmpeg process
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

while True:
    raw_frame = process.stdout.read(640 * 480 * 3)  # Read frame size

    if not raw_frame:
        print("Failed to read frame, retrying...")
        continue

    frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Convert to OpenCV format

    cv2.imshow("WEBCAM ", frame)

    print("Frame received:", frame.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()

