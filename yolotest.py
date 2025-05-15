import cv2
import torch
import numpy as np
import os
import subprocess
from ultralytics import YOLO
import time

HOME = os.path.expanduser("~")  # Gets /home/rasmusmade dynamically


print(HOME)

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = YOLO("yolo11s-seg.pt")
model.to("cuda")

rtspURL = "rtsp://100.103.175.86:8554/mystream?rtsp_transport=tcp"

# Run FFmpeg as a subprocess to read RTSP stream and output raw frames
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,
    "-r", "30",
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

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)

if cv2.getWindowProperty("Segmented Output", cv2.WND_PROP_VISIBLE) < 1:
    print("Window closed, reopening...")
    cv2.destroyAllWindows()  # Ensure no ghost windows
    cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)

try: 
    process.stdout.flush()
    while True:
        raw_frame = process.stdout.read(1280 * 720 * 3)  # Read frame size

        if not raw_frame:
            print("Failed to read frame, retrying...")
            continue
        
        if len(raw_frame) != (1280 * 720 * 3):
            print("Incomplete frame received, retrying...")
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            continue

        frame_counter += 1
        #if frame_counter % 3 == 0:
        #    flag = True

        if flag: 
            #frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Converting a 1D array into an image
            #frame = cv2.resize(frame, (1280, 720))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            inf_start = time.time()
            results = model(frame, save=False)
            #results = model.track(frame, persist=True, save=False)
            inf_end = time.time()
            
            
            inference_time = inf_end - inf_start
            total_inference_time += inference_time

            human_mask_overlay = np.zeros_like(frame)

            # Check if segmentation results exist
            if results[0].masks is not None and len(results[0].masks.data) > 0:
                classes = results[0].boxes.cls  # Tensor of class indices for each detection
                masks = results[0].masks.data   # Tensor of masks with shape: (N, height, width)
                for i, cls in enumerate(classes):
                    if int(cls) == 0:  # Assuming class index 0 corresponds to humans
                        # Convert mask from torch tensor to numpy array
                        mask = masks[i].cpu().numpy()
                        # Threshold the mask (assuming values between 0 and 1)
                        mask = (mask > 0.5).astype(np.uint8)
                        # Color the mask area red (BGR: (0, 0, 255))

                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        human_mask_overlay[mask == 1] = (0, 0, 255)


            # Blend the human mask overlay with the original frame
            alpha = 0.5
            blended_frame = cv2.addWeighted(frame, 1 - alpha, human_mask_overlay, alpha, 0)

            ### --- ADDED START ---
            current_time = time.time()
            elapsed = current_time - start_time
            fps = frame_counter / elapsed if elapsed > 0 else 0
            cv2.putText(blended_frame, f"{fps:.1f} FPS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 4)
            ### --- ADDED END ---

            cv2.imshow("Segmented Output", blended_frame)
            #flag = False

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