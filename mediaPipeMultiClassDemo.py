import cv2
import numpy as np
import os
import subprocess
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from screeninfo import get_monitors
import threading

# ==== Constants and Setup ====
#model_path = "/home/rasmus/thesisrepo/selfie_multiclass_256x256.tflite"
model_path = "/home/rasmus/thesisrepo/selfie_segmenter.tflite"

monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmented Output", SCREEN_WIDTH, SCREEN_HEIGHT)

# ==== Globals ====
global frame_rgb
latest_output_image = None
lock = threading.Lock()
frame_counter = 0
total_inference_time = 0.0
start_time = time.time()

# ==== Callback ====
def segmentation_result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global original_frame_rgb, latest_output_image

    category_mask = result.category_mask
    original_image = original_frame_rgb.copy()
    mask_np = category_mask.numpy_view()

    # Resize if needed
    if mask_np.shape != original_image.shape[:2]:
        mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Fixed color palette
    color_palette = {
        0: (0, 0, 0),         # background - black
        1: (128, 0, 128),     # hair - purple
        2: (255, 220, 180),   # body skin - light peach
        3: (255, 180, 150),   # face skin - peach
        4: (0, 128, 255),     # clothes - blue
        5: (255, 255, 0)      # others/accessories - yellow
    }

    # Colorize the mask
    color_mask = np.zeros_like(original_image)
    for class_id, color in color_palette.items():
        color_mask[mask_np == class_id] = color

    # Blend with original
    alpha = 0.5
    blended = cv2.addWeighted(original_image, 1 - alpha, color_mask, alpha, 0)

    with lock:
        latest_output_image = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

# ==== MediaPipe Setup ====
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_category_mask=True,
    result_callback=segmentation_result_callback
)

# ==== RTSP Stream Setup ====
rtspURL = ""

ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", rtspURL,
    "-r", "30",
    "-bufsize", "512k",
    "-f", "image2pipe",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an",
    "pipe:1"
]

process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

# ==== Display Function ====
def resize_and_show(image, fps):
    if image is None or image.size == 0:
        return
    resized = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    cv2.putText(resized, f"{fps:.1f} FPS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("Segmented Output", resized)

# ==== Main Loop ====
try:
    process.stdout.flush()
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        while True:
            raw_frame = process.stdout.read(1280 * 720 * 3)
            if len(raw_frame) != 1280 * 720 * 3:
                continue

            frame_counter += 1
            try:
                frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))
            except ValueError:
                continue

            original_frame_bgr = frame.copy()
            original_frame_rgb = cv2.cvtColor(original_frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(original_frame_rgb, (256, 256))

            inf_start = time.time()

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp = int(time.time() * 1000)
            segmenter.segment_async(image, timestamp)

            with lock:
                if latest_output_image is not None:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frame_counter / elapsed if elapsed > 0 else 0

                    resize_and_show(latest_output_image, fps)
                    latest_output_image = None

            inf_end = time.time()
            total_inference_time += (inf_end - inf_start)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    process.terminate()
    cv2.destroyAllWindows()

    end_time = time.time()
    total_runtime = end_time - start_time
    avg_inf_time = total_inference_time / frame_counter if frame_counter else 0

    print(f"Total frames: {frame_counter}")
    print(f"Runtime: {total_runtime:.2f}s")
    print(f"Avg inference/frame: {avg_inf_time:.4f}s ({1 / avg_inf_time:.2f} FPS)" if avg_inf_time > 0 else "No valid frames.")
