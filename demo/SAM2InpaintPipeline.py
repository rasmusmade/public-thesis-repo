import cv2
import torch
import sys
import os
import subprocess
import time
import queue

# -------------------------
# ADDING THE CORRECT PATH
# -------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
repoPath = os.path.join(HOME, "sam2repo", "sam2")  # NOT "sam2/sam2"
sys.path.append(repoPath)
print(f"sys.path includes: {repoPath}")
import numpy as np
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import pipeline_stop_event
torch.backends.cudnn.benchmark = True
import threading
from dotenv import load_dotenv

load_dotenv()
# -------------------------------
# DIMENSIONS AND OTHER VARIABLES
# ------------------------------
process = None

backgroundBuffer = None
inpaint_dilation_kernel = np.ones((5, 5), np.uint8)  # Dilate a bit to reduce edge artifacts

MODEL_W = int(os.getenv("MODEL_W", 512))
MODEL_H = int(os.getenv("MODEL_H", 512))
SCREEN_W = int(os.getenv("SCREEN_W", 1280))
SCREEN_H = int(os.getenv("SCREEN_H", 720))

#FRAME_HEIGHT = 480
#FRAME_WIDTH = 640
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
rtspURL = os.getenv("RTSP_URL")
click_override = False

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

# -------------------------
# GPU AND MODEL SETUP
# -------------------------

#torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join(HOME, "sam2repo", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
#CONFIG = os.path.join(HOME, "sam2repo", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
CONFIG = "configs/sam2.1/sam2.1_hiera_t"

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
sam2_model = torch.compile(sam2_model) # Compiling the model for speed 

predictor = SAM2ImagePredictor(sam2_model)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

# ────────────────────────  shared state  ────────────────────────
latest_jpeg      = None                 # newest encoded frame
jpeg_lock        = threading.Lock()

input_point      = np.array([[MODEL_W * 0.5, MODEL_H * 0.5]])
input_lock       = threading.Lock()     # protects input_point
input_label  = np.array([1], dtype=np.int64) 
# ────────────────────────────────────────────────────────────────

# -------------------------
# VIDEOEFEED SETUP
# -------------------------
# Run FFmpeg as a subprocess to read RTSP stream and output raw frames
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,
    "-vf", "fps=15", # HARD throttle: throw away frames in the demuxer 
    "-r", "15",
    "-bufsize", "512k",             # Input RTSP URL
    "-f", "image2pipe",        # Output as raw frames
    "-pix_fmt", "bgr24",       # Ensure OpenCV-compatible format
    "-vcodec", "rawvideo",     # Raw video format
    "-an",                     # No audio   # Resize if needed
    "pipe:1"                   # Output to pipe
]

# -------------------------
# CLICK EVENT SETUP
# -------------------------  

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

def set_click(x, y):
    global click_override

    with input_lock:
        input_point[:] = [[x, y]]
        click_override = True

FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3
frame_q = queue.Queue(maxsize=1)          # keeps only “latest”

def reader():
    """Read exactly one raw frame per iteration; drop older ones."""
    global process
    fd = process.stdout                         # this is already blocking

    while not pipeline_stop_event.is_set():
        raw = fd.read(FRAME_BYTES)              # ← BLOCKS until full frame
        if len(raw) != FRAME_BYTES:             # ffmpeg closed or short read
            break

        if frame_q.full():
            try:
                frame_q.get_nowait()            # discard stale frame
            except queue.Empty:
                pass
        frame_q.put_nowait(raw)

def _worker():
    global backgroundBuffer, click_override
    start_ffmpeg()
    threading.Thread(target=reader, daemon=True).start()

    fpsCounter, lastFpsUpdate = 0, time.time()
    fpsValue = 0
    
    while not pipeline_stop_event.is_set():
        try:
            raw = frame_q.get(timeout=0.2)
        except queue.Empty:
            continue

        native = np.frombuffer(raw, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
        small  = cv2.resize(native, (MODEL_W, MODEL_H))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ----- read current click safely -----
        with input_lock:
            pt = input_point.copy()
        # -------------------------------------

        predictor.set_image(rgb)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, _, _ = predictor.predict(
            point_coords=pt,
            point_labels=input_label,
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
                    with input_lock:
                        input_point[:] = [[cX, cY]]        # update
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

            
        # -------------------------------------
        # ─── draw click point ──────────────────────────────────────────
        x_scr = int(pt[0, 0] * SCREEN_W / MODEL_W)   # 512 → 1280
        y_scr = int(pt[0, 1] * SCREEN_H / MODEL_H)   # 512 →  720
        cv2.circle(native, (x_scr, y_scr), 5, (0, 255, 0), -1)
        # ───────────────────────────────────────────────────────────────

        # FPS counter
        fpsCounter += 1
        now = time.time()
        if now - lastFpsUpdate >= 1.0:
            fpsValue = fpsCounter
            fpsCounter = 0
            lastFpsUpdate = now
        
        text = f"FPS: {fpsValue}"
        cv2.putText(native, text, (fpsTextX, fpsTextY), textFont, textScale, textColor, textThickness)


        # encode & publish
        buf = cv2.imencode(".jpg", native)[1].tobytes()
        with jpeg_lock:
            global latest_jpeg
            latest_jpeg = buf

_worker_thread = None

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
    with jpeg_lock:
        return latest_jpeg