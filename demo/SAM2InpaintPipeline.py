import os
import sys
import cv2
import time
import queue
import torch
import threading
import numpy as np
import subprocess
from dotenv import load_dotenv
from utils import pipeline_stop_event

'''
PATH SETUP
'''
HOME = os.path.dirname(os.path.abspath(__file__))
repoPath = os.path.join(HOME, "sam2repo", "sam2")
sys.path.append(repoPath)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv

'''
ENV + CONSTANTS
'''
load_dotenv()

MODEL_W = int(os.getenv("MODEL_W", 512))
MODEL_H = int(os.getenv("MODEL_H", 512))
SCREEN_W = int(os.getenv("SCREEN_W", 1280))
SCREEN_H = int(os.getenv("SCREEN_H", 720))
RTSP_URL = os.getenv("RTSP_URL")

CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t"
CHECKPOINT_PATH = os.path.join(HOME, "sam2repo", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
FRAME_BYTES = SCREEN_W * SCREEN_H * 3

'''
MODEL + PREDICTOR SETUP
'''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sam2_model = torch.compile(build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE, apply_postprocessing=False))
predictor = SAM2ImagePredictor(sam2_model)
inpaint_dilation_kernel = np.ones((5, 5), np.uint8) 

'''
SHARED STATE
'''
process = None
latestJpeg = None                 # newest encoded frame
jpegLock = threading.Lock()

inputPoint = np.array([[MODEL_W * 0.5, MODEL_H * 0.5]])
inputLabel = np.array([1], dtype=np.int64) 
inputLock = threading.Lock()     # protects input_point

frameQueue = queue.Queue(maxsize=1)          # keeps only “latest”
click_override = False
backgroundBuffer = None
click_override = False
'''
FMMPEG COMMAND
'''
ffmpeg_cmd = [
    "ffmpeg", # Call the ffmpeg CLI tool
    "-rtsp_transport", "tcp", # Use tcp
    "-i", RTSP_URL, # Input URL
    "-vf", "fps=15", # Video filter to reduce it to 15FPS
    "-r", "15", # Ensures output of 15fps
    "-bufsize", "512k", # Buffer size to 512 kilobytes for smooth video delivery
    "-f", "image2pipe", # Output format: send frames are raw image stream to stdout       
    "-pix_fmt", "bgr24", # Use 24-bit BGR pixel format (compatible with OpenCV)      
    "-vcodec", "rawvideo", # Use raw video codec (no compression, suitable for real-time)     
    "-an", # Disable audio stream                     
    "pipe:1" # Output to pipe                  
]

'''
FPS OVERLAY
'''
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.2
TEXT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  
FPS_TEXT = "FPS: 99"
FPS_TEXT_SIZE = cv2.getTextSize(FPS_TEXT, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0] 
FPS_TEXT_X = SCREEN_W - FPS_TEXT_SIZE[0] - 20  # 20 px margin from right
FPS_TEXT_Y = 50  # 50 px from top

'''
HELPER FUNCTIONS
'''
def set_click(x, y):
    global click_override

    with inputLock:
        inputPoint[:] = [[x, y]]
        click_override = True

def start_ffmpeg():
    global process
    if process is None or process.poll() is not None:
        print("[SAM2 PIPELINE] Starting ffmpeg...")
        pipeline_stop_event.clear()
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

def stop_ffmpeg():
    global process
    pipeline_stop_event.set()
    if process is not None:
        print("[SAM2 PIPELINE] Stopping ffmpeg...")
        process.terminate()
        process.wait(timeout=2)
        process = None

def readFrames():
    """
    Read exactly one raw frame per iteration; drop older ones.
    """
    global process
    while not pipeline_stop_event.is_set():
        raw = process.stdout.read(FRAME_BYTES)
        if len(raw) != FRAME_BYTES:
            break
        if frameQueue.full():
            try: frameQueue.get_nowait()
            except queue.Empty: pass
        frameQueue.put_nowait(raw)

def _worker():
    """
    Grabs RTSP frames, runs SAM2 once per frame, publishes JPEG.
    """
    start_ffmpeg()
    threading.Thread(target=readFrames, daemon=True).start()

    fpsCounter, lastFpsUpdate, fpsValue = 0, time.time(), 0
    global backgroundBuffer, click_override, latestJpeg

    while not pipeline_stop_event.is_set():
        try:
            raw = frameQueue.get(timeout=0.2)
        except queue.Empty:
            continue

        native = np.frombuffer(raw, np.uint8).reshape((SCREEN_H, SCREEN_W, 3)).copy()
        small = cv2.resize(native, (MODEL_W, MODEL_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        with inputLock:
            pt = inputPoint.copy()

        predictor.set_image(rgb)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, _, _ = predictor.predict(
            point_coords=pt,
            point_labels=inputLabel,
            multimask_output=False
        )

        # ------- overlay red mask ------------
        if masks is not None and masks.shape[0] > 0:
            mask512 = masks[0].astype(np.uint8)
            mask720 = cv2.resize(mask512, (SCREEN_W, SCREEN_H),
                                 interpolation=cv2.INTER_NEAREST)

             # --- centroid auto‑tracking (512‑grid) ---
            if not click_override:
                m = cv2.moments(mask512)
                if m["m00"] != 0:
                    cX = int(m["m10"] / m["m00"])
                    cY = int(m["m01"] / m["m00"])
                    with inputLock:
                        inputPoint[:] = [[cX, cY]]        # update
            else:
                click_override = False

            # --------- inpainting on 1280×720 --------
            dilated = cv2.dilate(mask720, inpaint_dilation_kernel, iterations=1)
            dilated3 = np.repeat(dilated[:, :, None], 3, axis=2)

            if backgroundBuffer is None or backgroundBuffer.shape != native.shape:
                backgroundBuffer = native.copy().astype(np.float32)

            backgroundBuffer[:] = np.where(
                dilated3 == 0,
                0.97 * backgroundBuffer + 0.03 * native,
                backgroundBuffer)

            inpainted = np.where(dilated3 == 1,
                                backgroundBuffer.astype(np.uint8),
                                native)

            native = inpainted

        x_scr = int(pt[0, 0] * SCREEN_W / MODEL_W)   # 512 → 1280
        y_scr = int(pt[0, 1] * SCREEN_H / MODEL_H)   # 512 →  720
        cv2.circle(native, (x_scr, y_scr), 5, (0, 255, 0), -1)

        # FPS counter
        fpsCounter += 1
        now = time.time()
        if now - lastFpsUpdate >= 1.0:
            fpsValue = fpsCounter
            fpsCounter = 0
            lastFpsUpdate = now
        
        cv2.putText(native, f"FPS: {fpsValue}", (FPS_TEXT_X, FPS_TEXT_Y), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

        # encode & publish
        buf = cv2.imencode(".jpg", native)[1].tobytes()
        with jpegLock:
            latestJpeg = buf

_worker_thread = None

'''
CONTROL ENTRYPOINTS
'''
def start():
    """Spawn the worker if it isn’t running yet."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    pipeline_stop_event.clear()
    _worker_thread = threading.Thread(target=_worker, daemon=True)
    _worker_thread.start()

def stop():
    pipeline_stop_event.set()
    stop_ffmpeg()
    if _worker_thread:
        _worker_thread.join(timeout=1)

def get_latest_jpeg():
    with jpegLock:
        return latestJpeg

