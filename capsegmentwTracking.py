import cv2
import torch
import sys
import os
import subprocess

# -------------------------
# ADDING THE CORRECT PATH
# -------------------------
HOME = os.path.dirname(os.path.abspath(__file__))
repoPath = os.path.join(HOME, "mediapipe_demo", "sam2repo", "sam2")  # NOT "sam2/sam2"
sys.path.append(repoPath)
print(f"sys.path includes: {repoPath}")
import numpy as np
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from screeninfo import get_monitors

# -------------------------
# DIMENSIONS AND OTHER VARIABLES
# -------------------------

backgroundBuffer = None
inpaint_dilation_kernel = np.ones((9, 9), np.uint8)  # Dilate a bit to reduce edge artifacts
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height
#FRAME_HEIGHT = 480
#FRAME_WIDTH = 640
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
input_w, input_h = 512, 512
rtspURL = ""

# -------------------------
# GPU AND MODEL SETUP
# -------------------------

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = os.path.join(HOME, "mediapipe_demo", "sam2repo", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
CONFIG = os.path.join(HOME, "mediapipe_demo", "sam2repo", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
sam2_model = torch.compile(sam2_model) # Compiling the model for speed 

predictor = SAM2ImagePredictor(sam2_model)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

# -------------------------
# VIDEOEFEED SETUP
# -------------------------
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

# -------------------------
# CLICK EVENT SETUP
# -------------------------
input_point = np.array([[input_w*0.5, input_h*0.5]])  # Center of 320x240 image
input_label = np.array([1])
display_width = SCREEN_WIDTH
display_height = SCREEN_HEIGHT
model_input_width = input_w
model_input_height = input_h

def click_event(event, x, y, flags, param):
    global input_point, flag

    if event == cv2.EVENT_LBUTTONDOWN:
        # Map from display size (shown window) to model input size
        scaled_x = int(x * model_input_width / display_width)
        scaled_y = int(y * model_input_height / display_height)
        #scaled_x = x
        #scaled_y = y

        print(f"Clicked at ({x},{y}) → Model input: ({scaled_x},{scaled_y})")
        input_point = np.array([[scaled_x, scaled_y]])
        flag = True

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Segmented Output", SCREEN_WIDTH, SCREEN_HEIGHT)
cv2.namedWindow("Segmented Output", cv2.WINDOW_AUTOSIZE)  # remove cv2.WINDOW_NORMAL

cv2.setMouseCallback("Segmented Output", click_event)

frame_counter = 0
flag = True

while True:
    raw_frame = process.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)  # Read frame size

    if not raw_frame:
        print("Failed to read frame, retrying...")
        continue
    
    
    frame_counter += 1
     
    frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))  # Convert to OpenCV format
    frame = cv2.resize(frame, (input_w, input_h))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    # -------------------------------
    # Tracking logic
    # -------------------------------

    if masks is not None and masks.shape[0] > 0:
        mask = masks[0].astype(np.uint8)

        # Update tracking centroid
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            input_point = np.array([[cX, cY]])
            print(f"Tracking → New input_point: ({cX}, {cY})")
        else:
            print("Empty mask, retaining previous input_point")

        # === Inpainting starts here ===

        # Dilate mask to smooth edges and cover bleed
        expanded_mask = cv2.dilate(mask, inpaint_dilation_kernel, iterations=1)
        expanded_mask_3ch = np.repeat(expanded_mask[:, :, np.newaxis], 3, axis=2)

        # Initialize background buffer on first frame or if size changes
        if backgroundBuffer is None or backgroundBuffer.shape != frame.shape:
            backgroundBuffer = frame.copy().astype(np.float32)

        # Update background buffer only on background pixels
        backgroundBuffer = np.where(expanded_mask_3ch == 0,
                                    0.97 * backgroundBuffer + 0.03 * frame,
                                    backgroundBuffer)

        # Composite the image
        inpainted_frame = np.where(expanded_mask_3ch == 1,
                                backgroundBuffer.astype(np.uint8),
                                frame)

        output_to_show = inpainted_frame.copy()

    else:
        print("No mask detected, skipping centroid update")
        output_to_show = frame.copy()  # Show original frame if no mask

    # Optional: annotate detected masks
    #detections = sv.Detections(
    #    xyxy=sv.mask_to_xyxy(masks=masks),
    #    mask=masks.astype(bool)
    #)
    #output_to_show = mask_annotator.annotate(scene=output_to_show, detections=detections)

    # Draw clicked or tracked point
    for pt in input_point:
        cv2.circle(output_to_show, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    resized = cv2.resize(output_to_show, (SCREEN_WIDTH, SCREEN_HEIGHT))
    cv2.imshow("Segmented Output", resized)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()