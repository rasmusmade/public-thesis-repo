from flask import Flask, Response, render_template, request, jsonify
import pipeline
import threading
import cv2
import SAM2maskOnlyPipeline as Mask
import SAM2InpaintPipeline as Inp
import numpy as np   
import time   

app = Flask(__name__)
activePipeline = None
segmentationThread = None

def launch_mediapipe():
    global segmentationThread
    from utils import pipeline_stop_event
    pipeline_stop_event.set()          # 1) kill running pipeline
    Mask.stop()
    Inp.stop()
    pipeline.stop_ffmpeg()             # make sure both are down

    if segmentationThread and segmentationThread.is_alive():
        segmentationThread.join(timeout=1)  # wait a moment

    pipeline_stop_event.clear()             # we’re about to start fresh
    # 2) start the new one
    t = threading.Thread(target=pipeline.run_segmentation_loop, daemon=True)
    t.start()
    return t

@app.route('/start_mediapipe', methods=['POST'])
def start_mediapipe():
    global segmentationThread, activePipeline

    segmentationThread = launch_mediapipe()
    activePipeline = "mediapipe"
    return jsonify(success=True)

@app.route('/start_sam2mask', methods=['POST'])
def start_sam2mask():
    global activePipeline
    Inp.stop()         # stop the other SAM pipeline
    pipeline.stop_ffmpeg()      # stop mediapipe
    Mask.start()       # <<< start the worker
    activePipeline = "sam2mask"
    return jsonify(success=True)


@app.route('/start_sam2inpaint', methods=['POST'])
def start_sam2inpaint():
    global activePipeline
    Mask.stop()    # stop the other SAM pipeline 
    pipeline.stop_ffmpeg()
    Inp.start()
    activePipeline = "sam2inpaint"
    return jsonify(success=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/current_pipeline')
def current_pipeline():
    return jsonify(pipeline=activePipeline or "none")


@app.route('/video_feed')
def video_feed():
    def stream():
        SLEEP = 1/30
        while True:
            if activePipeline == "mediapipe":
                with pipeline.lock:
                    img = pipeline.latest_output_image
                if img is None:
                    time.sleep(0.01); continue
                ok, buf = cv2.imencode('.jpg', img)
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                    time.sleep(SLEEP)

            elif activePipeline == "sam2mask":
                    buf = Mask.get_latest_jpeg()
                    if buf is not None:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'
                    time.sleep(SLEEP)    # limit push rate


            elif activePipeline == "sam2inpaint":
                    buf = Inp.get_latest_jpeg()
                    if buf is not None:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'
                    time.sleep(SLEEP)    # limit push rate

            else:
                time.sleep(0.05)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click', methods=['POST'])
def click():
    data = request.get_json()
    x, y = int(data['x']), int(data['y'])
    Mask.set_click(x, y)
    Inp.set_click(x, y)     
    return '', 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
