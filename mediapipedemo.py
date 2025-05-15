import cv2
from cv2 import imshow
import torch
import numpy as np
import os
import subprocess
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
model_path = "/home/rasmusmade/segment-anything-2/selfie_segmenter.tflite"
from PIL import Image
from screeninfo import get_monitors

monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmented Output", SCREEN_WIDTH, SCREEN_HEIGHT)

global frame_rgb 
latest_output_image = None

def segmentation_result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global original_frame_rgb, latest_output_image

    category_mask = result.category_mask
    original_image = original_frame_rgb.copy()
    mask_np = category_mask.numpy_view()

    print(f"Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    print(f"Mask min: {np.min(mask_np)}, max: {np.max(mask_np)}, mean: {np.mean(mask_np)}")

    # Resize if needed
    if mask_np.shape != original_image.shape[:2]:
        mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to binary mask (fixed threshold for uint8 mask)
    binary_mask = (mask_np <= 127).astype(np.uint8)

    # Expand to 3 channels
    binary_mask_3ch = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    # Create overlay
    overlay_color = np.array([255, 0, 0], dtype=np.float32)  # Red in RGB
    overlay = np.ones_like(original_image, dtype=np.float32) * overlay_color

    # Prepare original
    original_image = original_image.astype(np.float32)

    # Blend
    alpha = 0.5
    blended = np.where(
        binary_mask_3ch == 1,
        overlay * alpha + original_image * (1 - alpha),
        original_image
    )

    # Convert to uint8 for display
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    latest_output_image = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    print(f"[{timestamp_ms}] Got segmentation result!")


def print_result(result, output_image: Image, timestamp_ms: int):
    print('segmented masks size: {}'.format(len(result)))

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

print(HOME)

#torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

#if torch.cuda.get_device_properties(0).major >= 8:
#    torch.backends.cuda.matmul.allow_tf32 = True
#    torch.backends.cudnn.allow_tf32 = True

rtspURL = ""

# Run FFmpeg as a subprocess to read RTSP stream and output raw frames
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,
    "-r", "20",
    "-bufsize", "512k",             # Input RTSP URL
    "-f", "image2pipe",        # Output as raw frames
    "-pix_fmt", "bgr24",       # Ensure OpenCV-compatible format
    "-vcodec", "rawvideo",     # Raw video format
    "-an",                     # No audio   # Resize if needed
    "pipe:1"                   # Output to pipe
]

# Start FFmpeg process
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
flag = True

#mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
# frame_width, frame_height = 320, 240
frame_counter = 0
total_inference_time = 0.0
start_time = time.time()

model_path = "/home/rasmusmade/segment-anything-2/selfie_segmenter.tflite"

base_options = BaseOptions(model_asset_path=model_path)


#cv2.setWindowProperty("Segmented Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

last_win_size = [800, 600]

def resize_and_show(image):
    if image is None or image.size == 0:
        print("Invalid image.")
        return
        
    resized = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    cv2.imshow("Segmented Output", resized)

try:
    process.stdout.flush()
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        while True:
            raw_frame = process.stdout.read(1280 * 720 * 3)  # Read frame size
            

            print(f"Read {len(raw_frame)} bytes from FFmpeg")
            if len(raw_frame) != 1280 * 720 * 3:
                print("Warning: Incomplete frame received!")

            if not raw_frame:
                print("Failed to read frame, retrying...")
                continue
            
            frame_counter += 1
            #if frame_counter % 3 == 0:
            #    flag = True

            if flag: 
                frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))  # Converting a 1D array into an image
                #frame = cv2.resize(frame, (1280, 720))
                original_frame_bgr = frame.copy()

                original_frame_rgb = cv2.cvtColor(original_frame_bgr, cv2.COLOR_BGR2RGB)

                
                frame_rgb = cv2.resize(original_frame_rgb, (256, 256))

                inf_start = time.time()

                            # Create the MediaPipe image file that will be segmented
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                print("the image part was successsful!")

                
                timestamp = int(time.time() * 1000)
                # Retrieve the masks for the segmented image
                segmenter.segment_async(image, timestamp)
                
                if latest_output_image is not None:
                    resize_and_show(latest_output_image)
                    cv2.waitKey(1)
                    latest_output_image = None

                
                #results = model.track(frame, persist=True, save=False)
                inf_end = time.time()
                
                
                inference_time = inf_end - inf_start
                total_inference_time += inference_time


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nInterrupted! Calculating final stats...")

finally:
    process.terminate()

    end_time = time.time()
    total_runtime = end_time - start_time
    average_inference_time = total_inference_time / frame_counter if frame_counter > 0 else 0

    print(f"Total frames processed: {frame_counter}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time per frame: {average_inference_time:.4f} seconds ({1/average_inference_time:.2f} FPS)" if average_inference_time > 0 else "No valid frames processed.")