from flask import Flask, Response, render_template
import threading
import cv2
import numpy as np

app = Flask(__name__)
lock = threading.Lock()
latest_output_image = None  # Updated by your segmentation code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_output_image
        while True:
            with lock:
                if latest_output_image is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_output_image)
                    if not ret:
                        continue
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
