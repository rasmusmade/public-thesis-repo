import cv2
import torch
import base64
import os
import subprocess

HOME = os.getcwd()

print(HOME)

import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"{HOME}/checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
sam2_model = torch.compile(sam2_model)
#mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

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
frame_width, frame_height = 320, 240
box_width, box_height = 50, 50

center_x = frame_width // 2
center_y = frame_height // 2

single_box = np.array([[
    center_x - box_width // 2,
    center_y - box_height // 2,
    center_x + box_width // 2,
    center_y + box_height // 2
]])

boxes = single_box

input_point = np.array([
    [
        (box[0] + box[2]) // 2,  # x center
        (box[1] + box[3]) // 2   # y center
    ] for box in boxes
])

input_label = np.ones(input_point.shape[0])

def click_event(event, x, y, flags, param):
    global input_point, flag

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"New point clicked: {x}, {y}")
        input_point = np.array([[x,  y]])
        flag = True

predictor = SAM2ImagePredictor(sam2_model)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
frame_counter = 0

cv2.namedWindow("Segmented Output")
cv2.setMouseCallback("Segmented Output", click_event)

while True:
    raw_frame = process.stdout.read(640 * 480 * 3)  # Read frame size

    if not raw_frame:
        print("Failed to read frame, retrying...")
        continue
    
    frame_counter += 1

    if frame_counter % 3 == 0:
        flag = True

    if flag: 
        frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Convert to OpenCV format
        frame = cv2.resize(frame, (320, 240))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predictor.set_image(image_rgb)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        if boxes.shape[0] != 1:
            masks = np.squeeze(masks)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask = masks.astype(bool)
        )

        annotated_image = mask_annotator.annotate(scene=frame.copy(), detections=detections)

        # Drawing the clicked point
        for pt in input_point:
            cv2.circle(annotated_image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)


        cv2.imshow("Segmented Output", annotated_image)
        flag = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()