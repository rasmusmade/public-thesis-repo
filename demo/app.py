from flask import Flask, Response, render_template, request, jsonify
import threading
import cv2
import numpy as np   
import time   
import SAM2maskOnlyPipeline as Mask
import SAM2InpaintPipeline as Inp
import MediapipePipeline

app = Flask(__name__)

'''
Globals for tracking which pipeline is currently active and its thread.
'''
activePipeline = None
segmentationThread = None


def launch_mediapipe():
    """
    Stop any running pipelines and then start the MediaPipe segmentation pipeline in a new background thread.
    """
    global segmentationThread
    from utils import pipeline_stop_event

    pipeline_stop_event.set()          # Set the stop event to to stop any running pipeline
    Mask.stop()
    Inp.stop()
    MediapipePipeline.stop()             

    if segmentationThread and segmentationThread.is_alive():
        segmentationThread.join(timeout=1)  # Waiting for the previous segmentation thread to finish 

    pipeline_stop_event.clear() # Clear the flag so the running pipeline won't stop          
    t = threading.Thread(target=MediapipePipeline.run_segmentation_loop, daemon=True) # Starting the Mediapipe pipeline in a new thread
    t.start()
    return t

@app.route('/start_mediapipe', methods=['POST'])
def start_mediapipe():
    """
    HTTP endpoint to start the MediaPipe pipeline.
    Stops any other running pipelines first.
    """
    global segmentationThread, activePipeline

    segmentationThread = launch_mediapipe() # Setting the globals correctly
    activePipeline = "mediapipe"
    return jsonify(success=True)

@app.route('/start_sam2mask', methods=['POST'])
def start_sam2mask():
    """
    HTTP endpoint to start the sam2 mask pipeline.
    Stops any other running pipelines first.
    """
    global activePipeline
    Inp.stop()         # Stop the other SAM pipeline
    MediapipePipeline.stop()      # Stop mediapipe
    Mask.start()       # Start the sam2 mask pipeline
    activePipeline = "sam2mask"
    return jsonify(success=True)


@app.route('/start_sam2inpaint', methods=['POST'])
def start_sam2inpaint():
    """
    HTTP endpoint to start the sam2 inpaint pipeline.
    Stops any other running pipelines first.
    """
    global activePipeline
    Mask.stop()    # Stop the other SAM pipeline 
    MediapipePipeline.stop()
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
                with MediapipePipeline.lock:
                    img = MediapipePipeline.latest_output_image
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
