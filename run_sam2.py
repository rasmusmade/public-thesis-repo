import torch
from sam2.build_sam import build_sam2_video_predictor
import supervision as sv # Library for video frame extraction, annotation, and video processing
import numpy as np
import cv2

# https://blog.roboflow.com/sam-2-video-segmentation/
# Device and model setup

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Checkpoint and configuration of the specific model
CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Loading the model
sam2Model = build_sam2_video_predictor(CONFIG, CHECKPOINT, DEVICE)


# Reads the video and extracts frames
framesGenerator = sv.get_video_frames_generator("/home/rasmusmade/segment-anything-2/vecteezy_game-basketball-scores_1626739.mov")

sink = sv.ImageSink(
    target_dir_path="basketBallFrames",
    image_name_pattern="{:05d}.jpeg")

first_frame_path = None

# Saves the frames one by one and saves the frame on which i wish to select the object that I need segmented
with sink:
    for idx, frame in enumerate(framesGenerator):
        sink.save_image(frame)
        if idx == 10:
            first_frame_path = f"basketBallFrames/{idx:05d}.jpeg"

# Selecting the trackable object
first_frame = cv2.imread(first_frame_path)

clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Selected point: {x}, {y}")
        cv2.destroyAllWindows()

cv2.imshow("Select Object", first_frame)
cv2.setMouseCallback("Select Object", click_event)
cv2.waitKey(0)

if not clicked_points:
    raise RuntimeError("No point selected!")

points = np.array([clicked_points[0]], dtype=np.float32)
labels = np.array([1]) # Positive click
frame_idx = 10
tracker_id = 1

# Inference
inference_state = sam2Model.init_state("basketBallFrames")

_, object_ids, mask_logits = sam2Model.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=tracker_id,
    points=points,
    labels=labels,
)

colors = ['#FF1493']
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK)

video_info = sv.VideoInfo.from_video_path("/home/rasmusmade/segment-anything-2/vecteezy_game-basketball-scores_1626739.mov")

frames_paths = sorted(sv.list_files_with_extensions(
    directory="basketBallFrames", 
    extensions=["jpeg"]))

with sv.VideoSink("basketBallTracked.mp4", video_info=video_info) as sink:
    for frame_idx, object_ids, mask_logits in sam2Model.propagate_in_video(inference_state):
        frame = cv2.imread(frames_paths[frame_idx]) 
        masks = (mask_logits > 0.0).cpu().numpy()
        N, X, H, W = masks.shape
        masks = masks.reshape(N * X, H, W)
        detections = sv.Detections( # Converts the predicted mask into a bounding bix and mask
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=np.array(object_ids)
        )
        frame = mask_annotator.annotate(frame, detections) # Draws the segmentation mask
        sink.write_frame(frame)


