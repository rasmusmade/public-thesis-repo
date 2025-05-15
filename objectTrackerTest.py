import os
import time
import urllib
import subprocess

import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image

from sam2.build_sam import build_sam2_object_tracker

HOME = os.getcwd()

class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 ):
        
        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )
        
        return mask

    def add_frame(self, frame, mask):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()
        
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]
                
        rgb_frame = Image.fromarray(frame)
        clear_output(wait=True)
        display(rgb_frame)

NUM_OBJECTS = 2
sam2_checkpoint = f"{HOME}/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEVICE = 'cuda:0'

# Video capturing part
rtspURL = ""
ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",  # Force TCP
    "-i", rtspURL,
    "-r", "20",
    "-bufsize", "512k",
    "-f", "image2pipe",
    "-pix_fmt", "bgr24", # The outputted frame is in bgr
    "-vcodec", "rawvideo",
    "-an",
    "pipe:1",
    "-s", "320x240"
]

process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

video_height = 480
video_width = 640

Visualizer = Visualizer(
    video_width=video_width,
    video_height=video_height
)

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=model_cfg,
                                ckpt_path=sam2_checkpoint,
                                device=DEVICE,
                                verbose=False
                            )

available_slots = np.inf

first_frame = True
with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while True:
        start_time = time.time()
        frame = process.stdout.read(640 * 480 * 3)
        frame = np.frombuffer(frame, np.uint8).reshape((480, 640, 3))
        frame = cv2.resize(frame, (320, 240))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if first_frame:
            
# Read and process the first frame
raw_frame = process.stdout.read(640 * 480 * 3)
if not raw_frame:
    print("Error: Unable to read the first frame.")
    process.terminate()
    exit(1)

frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))
frame = cv2.resize(frame, (320, 240)) 