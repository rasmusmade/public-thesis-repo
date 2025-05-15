import cv2
import torch
import sys
import os
import subprocess
import numpy as np
import supervision as sv
HOME = os.path.dirname(os.path.abspath(__file__))
repoPath = os.path.join(HOME, "sam2repo", "sam2")  # NOT "sam2/sam2"
sys.path.append(repoPath)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ----------------------
# Setup your model here
# ----------------------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
CHECKPOINT = os.path.join(HOME, "sam2repo", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
CONFIG = os.path.join(HOME, "sam2repo", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_w, input_h = 512, 512  # Model input size
display_w, display_h = 1280, 720  # We display and also read frames at this resolution

# Example placeholders
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
sam2_model = torch.compile(sam2_model)
predictor = SAM2ImagePredictor(sam2_model)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

# --------------------------------
# Subprocess to read RTSP frames
# --------------------------------
rtspURL = "rtsp://172.17.154.156:8554/mystream?rtsp_transport=tcp"
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", rtspURL,
    "-r", "20",
    "-bufsize", "512k",
    "-f", "image2pipe",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an",
    "pipe:1"
]
process = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=10**8
)

# -----------------------------------
# Click callback and coordinate logic
# -----------------------------------
# We'll store the click in model coordinates:
input_point = np.array([[input_w*0.5, input_h*0.5]])  # center by default
input_label = np.array([1])

def click_event(event, x, y, flags, param):
    global input_point
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert from 1280×720 display → 512×512 model
        scale_x = input_w / display_w   # 512/1280
        scale_y = input_h / display_h   # 512/720
        model_x = int(x * scale_x)
        model_y = int(y * scale_y)
        print(f"Clicked at ({x},{y}) → Model input: ({model_x},{model_y})")
        input_point = np.array([[model_x, model_y]])

cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmented Output", display_w, display_h)
cv2.setMouseCallback("Segmented Output", click_event)

while True:
    raw_frame = process.stdout.read(display_w * display_h * 3)
    if not raw_frame:
        print("Failed to read frame, retrying...")
        continue
    
    # Convert from raw bytes to an OpenCV frame of size (720, 1280)
    frame = np.frombuffer(raw_frame, np.uint8).reshape((display_h, display_w, 3))

    # ------------------------------------------------
    # 1) Keep a copy for display
    # ------------------------------------------------
    display_frame = frame.copy()  # shape (720,1280)

    # ------------------------------------------------
    # 2) Resize to 512×512 for the model
    # ------------------------------------------------
    model_frame = cv2.resize(frame, (input_w, input_h))
    image_rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------
    # 3) Pass to your SAM2 predictor
    # ------------------------------------------------
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    # ------------------------------------------------
    # 4) Convert model’s 512×512 results back to 1280×720
    # ------------------------------------------------
    # Create Detections with masks in 512×512 domain
    detections_512 = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks.astype(bool)
    )

    # Upscale detection coordinates and masks to 1280×720
    detections_720 = detections_512._apply_scaling(
        x_scale=display_w / input_w,  # 1280/512
        y_scale=display_h / input_h,  # 720/512
    )

    # ------------------------------------------------
    # 5) Annotate on the 1280×720 "display_frame"
    # ------------------------------------------------
    annotated_image = mask_annotator.annotate(
        scene=display_frame,  # shape (720,1280)
        detections=detections_720
    )

    # Optionally draw the click point (which is still in 512×512).
    # First, scale that click to 1280×720 so it lines up visually:
    for pt in input_point:
        disp_x = int(pt[0] * (display_w / input_w))
        disp_y = int(pt[1] * (display_h / input_h))
        cv2.circle(annotated_image, (disp_x, disp_y), 5, (0, 255, 0), -1)

    cv2.imshow("Segmented Output", annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()
