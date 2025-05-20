import cv2
import numpy as np
import os
import subprocess
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv
import threading
from utils import pipeline_stop_event
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(BASE_DIR, "selfie_segmenter.tflite")
#model_path = default_model_path
model_path = os.getenv("MODEL_PATH", default_model_path)

rtspURL = os.getenv("RTSP_URL")
SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH", "1280"))
SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT", "720"))
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

print("[INFO] Model path in container:", model_path)
print("[INFO] Model file found:", os.path.exists(model_path))
print("[INFO] feed URL from env file:", rtspURL)
global frame_rgb 
fpsCounter = 0
fpsValue = 0
lastFpsUpdateTime = time.time()
isSegmenting = False
latest_output_image = None
backgroundBuffer = None 
lock = threading.Lock() # So that the callback function and main loop wouldn't try to access the latest_output_image at the same time

# -------------------------------
# FPS text size
# ------------------------------
# Text appearance settings
textFont = cv2.FONT_HERSHEY_SIMPLEX
textScale = 1.2
textThickness = 2
textColor = (0, 0, 255)  # Blue

# Precompute text size with widest expected text (e.g. "FPS: 99")
maxText = "FPS: 99"
fpsTextSize = cv2.getTextSize(maxText, textFont, textScale, textThickness)[0] 

# Hardcoded top-right position (1280 px width frame)
fpsTextX = 1280 - fpsTextSize[0] - 20  # 20 px margin from right
fpsTextY = 50  # 50 px from top


MODEL_INPUT_SIZE = (256, 256)

process = None 

def stop():
    from utils import pipeline_stop_event
    global process
    pipeline_stop_event.set()
    if process is not None:
        print("[MediaPipe PIPELINE] Stopping ffmpeg...")
        process.terminate()
        process.wait(timeout=2)
        process = None

def segmentation_result_callback(result, outputImage: mp.Image, timestamp_ms: int):
    global original_frame_rgb, latest_output_image, backgroundBuffer, isSegmenting

    isSegmenting = False
    category_mask = result.category_mask
    original_image = original_frame_rgb.copy()
    mask_np = category_mask.numpy_view()

    # Resize mask to match frame size
    if mask_np.shape != original_image.shape[:2]:
        mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to binary mask: person = 1, background = 0
    binary_mask = (mask_np <= 127).astype(np.uint8)

    # === Expand edges slightly with dilation
    kernelSize = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
    #kernel = np.ones((21, 21), np.uint8)  # 3-7 works well
    expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Repeat for 3 channels
    expanded_mask_3ch = np.repeat(expanded_mask[:, :, np.newaxis], 3, axis=2)

    # === Background logic ===
    if backgroundBuffer is None or backgroundBuffer.shape != original_image.shape:
        backgroundBuffer = original_image.copy()

    # Running average only for non-masked (background) regions
    backgroundBuffer = np.where(expanded_mask_3ch == 0,
                                0.97 * backgroundBuffer + 0.03 * original_image, # the smaller the value the less ghosting, but when too small the background may feel like its lagging
                                backgroundBuffer)

    # Composite output: show background where mask is 1
    outputImage = np.where(expanded_mask_3ch == 1, backgroundBuffer, original_image)
    outputImage = cv2.resize(outputImage, (SCREEN_WIDTH, SCREEN_HEIGHT))

    global fpsCounter, fpsValue, lastFpsUpdateTime

    fpsCounter += 1
    currentTime = time.time()

    if currentTime - lastFpsUpdateTime >= 1.0:
        fpsValue = fpsCounter
        fpsCounter = 0
        lastFpsUpdateTime = currentTime
    
    text = f"FPS: {fpsValue}"
    cv2.putText(outputImage, text, (fpsTextX, fpsTextY), textFont, textScale, textColor, textThickness)

    with lock: # Threading lock
        
        latest_output_image = cv2.cvtColor(outputImage.astype(np.uint8), cv2.COLOR_RGB2BGR)
      
        #print("Updated the latest output image")
    #print(f"[{timestamp_ms}] Segmentation + inpainting complete.")

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_path=model_path)

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_category_mask=True,
    result_callback=segmentation_result_callback  # <- REQUIRED for live mode
)


HOME = os.path.expanduser("~")  # Gets /home/rasmusmade dynamically

# Run FFmpeg as a subprocess to read RTSP stream and output raw frames
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,
    "-vf", "fps=25",
    "-r", "25",
    "-bufsize", "512k",             # Input RTSP URL
    "-f", "image2pipe",        # Output as raw frames
    "-pix_fmt", "bgr24",       # Ensure OpenCV-compatible format
    "-vcodec", "rawvideo",     # Raw video format
    "-an",                     # No audio   # Resize if needed
    "pipe:1"                   # Output to pipe
]
frame_counter = 0
total_inference_time = 0.0
start_time = time.time()

base_options = BaseOptions(model_asset_path=model_path)

#cv2.setWindowProperty("Segmented Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

last_win_size = [800, 600]
frames_skipped = 0

def run_segmentation_loop():
    print("!!!!!!!!!! run_segmentation_loop has started.")
    from utils import pipeline_stop_event
    global latest_output_image, backgroundBuffer, isSegmenting, process
    pipeline_stop_event.clear()
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    frame_counter = 0
    total_inference_time = 0.0
    start_time = time.time()

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        try:
            while not pipeline_stop_event.is_set():
                raw_frame = process.stdout.read(1280 * 720 * 3)
                if len(raw_frame) != 1280 * 720 * 3:
                    #print("Skipped a frame")
                    continue

                frame_counter += 1
                try:
                    frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))
                except ValueError:
                    continue

                original_frame_bgr = frame.copy()
                original_frame_rgb = cv2.cvtColor(original_frame_bgr, cv2.COLOR_BGR2RGB)
                globals()['original_frame_rgb'] = original_frame_rgb
                frame_rgb = cv2.resize(original_frame_rgb, (256, 256))

                timestamp = time.perf_counter_ns() // 1_000
                
                if not isSegmenting:
                    isSegmenting = True
                    segmenter.segment_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp)

                time.sleep(0.001)  # Keep CPU usage sane
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            process.terminate()

