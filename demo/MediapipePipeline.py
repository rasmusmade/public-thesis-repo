import os
import cv2
import time
import numpy as np
import threading
import subprocess
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks.python import vision
from utils import pipeline_stop_event

# Load env variables
load_dotenv()

# Constants and config
SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH", "1280"))
SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT", "720"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "selfie_segmenter.tflite"))
RTSP_URL = os.getenv("RTSP_URL")
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Print statemets for the terminal
print("[INFO] Model path in container:", MODEL_PATH)
print("[INFO] Model file found:", os.path.exists(MODEL_PATH))

# FPS Overlay settings
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.2
TEXT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)  # Red
FPS_TEXT = "FPS: 99"
FPS_TEXT_SIZE = cv2.getTextSize(FPS_TEXT, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0] 
FPS_TEXT_X = 1280 - FPS_TEXT_SIZE[0] - 20  # 20 px margin from right
FPS_TEXT_Y = 50  # 50 px from top

# Globals
frameRGB = None
fpsCounter = 0
fpsValue = 0
lastFpsUpdateTime = time.time()
isSegmenting = False
latest_output_image = None
backgroundBuffer = None 
lock = threading.Lock()
process = None

# Ffmpeg command for RTSP frames
FFMPEG_CMD = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", RTSP_URL,
    "-vf", "fps=25",
    "-r", "25",
    "-bufsize", "512k",             # Input RTSP URL
    "-f", "image2pipe",        # Output as raw frames
    "-pix_fmt", "bgr24",       # Ensure OpenCV-compatible format
    "-vcodec", "rawvideo",     # Raw video format
    "-an",                     # No audio   # Resize if needed
    "pipe:1"                   # Output to pipe
]

def stop():
    """
    Stop segmentation loop and terminate FFmpeg
    """
    global process
    pipeline_stop_event.set()
    if process is not None:
        print("[MediaPipe PIPELINE] Stopping ffmpeg...")
        process.terminate()
        process.wait(timeout=2)
        process = None

def segmentation_result_callback(result, image, timestamp_ms):
    """
    Callback function for MediaPipe segmentation results
    """
    global latest_output_image, backgroundBuffer, isSegmenting, fpsCounter, fpsValue, lastFpsUpdateTime, frameRGB

    isSegmenting = False
    category_mask = result.category_mask
    original = frameRGB.copy()
    mask_np = category_mask.numpy_view()

    # Resize mask to match frame size
    if mask_np.shape != original.shape[:2]:
        mask_np = cv2.resize(mask_np, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    binary_mask = (mask_np <= 127).astype(np.uint8) # Convert to binary mask: person = 1, background = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)) # Expand edges slightly with dilation
    expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    expanded_mask_3ch = np.repeat(expanded_mask[:, :, np.newaxis], 3, axis=2) # Repeat for 3 channels

    if backgroundBuffer is None or backgroundBuffer.shape != original.shape: # Initial frame
        backgroundBuffer = original.copy()

    backgroundBuffer = np.where(
        expanded_mask_3ch == 0,
        0.97 * backgroundBuffer + 0.03 * original, # 3% update on each frame
        backgroundBuffer)

    outputImage = np.where(expanded_mask_3ch == 1, backgroundBuffer, original) # Composite output: show background where mask is 1
    outputImage = cv2.resize(outputImage, (SCREEN_WIDTH, SCREEN_HEIGHT))

    ###
    # Adding the FPS text on the frame
    ###
    fpsCounter += 1
    currentTime = time.time()
    if currentTime - lastFpsUpdateTime >= 1.0:
        fpsValue = fpsCounter
        fpsCounter = 0
        lastFpsUpdateTime = currentTime
    
    text = f"FPS: {fpsValue}"
    cv2.putText(outputImage, text, (FPS_TEXT_X, FPS_TEXT_Y), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    with lock: # Threading lock
        latest_output_image = cv2.cvtColor(outputImage.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Mediapipe setup

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = vision.RunningMode

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_category_mask=True,
    result_callback=segmentation_result_callback 
)

def run_segmentation_loop():
    '''
    Main loop for processing frames from RTSP feed and segmenting them with Mediapipe SelfieSegmenter
    '''

    print("[MediaPipe] Segmentation loop started.")
    global latest_output_image, backgroundBuffer, isSegmenting, process, frameRGB
    pipeline_stop_event.clear()
    process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8) # Starting the FFMPEG subprocess

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        while not pipeline_stop_event.is_set():
            rawFrame = process.stdout.read(1280 * 720 * 3)
            if len(rawFrame) != 1280 * 720 * 3:
                continue

            try:
                frame = np.frombuffer(rawFrame, np.uint8).reshape((720, 1280, 3))
            except ValueError:
                continue

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameResized = cv2.resize(frameRGB, (256, 256))
            timestamp = time.perf_counter_ns() // 1_000
            
            if not isSegmenting:
                isSegmenting = True
                segmenter.segment_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frameResized), timestamp)

            time.sleep(0.001)  

def get_latest_output_image():
    """
    Thread-safe getter for the latest output image.
    """
    with lock:
        return latest_output_image

