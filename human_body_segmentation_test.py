import cv2
import torch
import numpy as np
import os
import subprocess
import time
from PIL import Image
from screeninfo import get_monitors
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import matplotlib.pyplot as plt

# ================== SETTINGS ==================
rtspURL = ""
model_path = "/home/rasmusmade/segment-anything-2/human-body-segmentation/body-seg-IIG.pth"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmented Output", SCREEN_WIDTH, SCREEN_HEIGHT)

# ================== LOAD MODEL ==================
print("Loading model...")
model = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
model.eval()
device = next(model.parameters()).device
print("Model loaded on:", device)

# ================== LOAD FEATURE EXTRACTOR ==================
extractor = SegformerFeatureExtractor(do_resize=True, size=512, do_normalize=True)

# ================== START FFMPEG PROCESS ==================
ffmpeg_cmd = [
    "ffmpeg", "-rtsp_transport", "tcp", "-i", rtspURL,
    "-r", "20", "-bufsize", "512k",
    "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo",
    "-an", "pipe:1"
]
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

# ================== PROCESS LOOP ==================
frame_counter = 0
total_inference_time = 0.0
start_time = time.time()

try:
    while True:
        raw_frame = process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
        if not raw_frame or len(raw_frame) != (FRAME_WIDTH * FRAME_HEIGHT * 3):
            print("Incomplete or empty frame, retrying...")
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        frame_counter += 1

        if frame_counter % 3 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Preprocess
            inputs = extractor(images=pil_img, return_tensors="pt").to(device)

            # Inference
            inf_start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # Shape: [1, num_classes, H, W]
                mask = torch.argmax(logits, dim=1)[0].cpu().numpy()  # Shape: [H, W]
            inf_end = time.time()
            total_inference_time += (inf_end - inf_start)
            frame_counter += 1
            
            unique_vals = np.unique(mask)
            print("Unique mask values:", unique_vals)
            # Resize mask to original frame size

            COLORMAP = (plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8).tolist()

            mask_resized = cv2.resize(mask.astype(np.uint8), (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            color_overlay = np.zeros_like(frame)
            for class_id in np.unique(mask_resized):
                if class_id < len(COLORMAP):
                    color_overlay[mask_resized == class_id] = COLORMAP[class_id]
                else:
        # If more classes than colors defined, just cycle through
                    color_overlay[mask_resized == class_id] = COLORMAP[class_id % len(COLORMAP)]   # Red color for class 1 (humans)

            # Blend overlay with original
            alpha = 0.5
            blended = cv2.addWeighted(frame, 1 - alpha, color_overlay, alpha, 0)

            # Show result
            resized_display = cv2.resize(blended, (SCREEN_WIDTH, SCREEN_HEIGHT))
            cv2.imshow("Segmented Output", resized_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    process.terminate()
    cv2.destroyAllWindows()

    end_time = time.time()
    runtime = end_time - start_time
    avg_inf_time = total_inference_time / frame_counter if frame_counter > 0 else 0

    print(f"Frames processed: {frame_counter}")
    print(f"Total runtime: {runtime:.2f}s")
    print(f"Average inference time per frame: {avg_inf_time:.3f}s ({1/avg_inf_time:.2f} FPS)" if avg_inf_time > 0 else "No valid frames.")
