# Real-Time Object Selection, Tracking, and Replacement with Background

This repository contains the practical part of my bachelor's thesis, which focuses on the real-time segmentation and removal of human subjects from a live video feed using deep learning models and background inpainting techniques.

## Abstract:
With the development of deep learning-based models, image and video segmentation has
made a significant leap forward. The aim of this bachelor's thesis was to create a real-time
video stream segmentation demo that detects a person in the image and replaces them with
background pixels. During the work, various modern deep learning-based segmentation
models were tested, and the most suitable ones were selected for the final demo. The purpose
of the completed demo was to introduce the capabilities of modern segmentation models to
Delta visitors. An analysis based on user feedback revealed that for the demo to function
more effectively, a computer with high hardware capacity is required.

This repository contains the code that was used in the final version of the demo. 

## Hardware and software requirements
An NVIDIA GPU (e.g., RTX 3080 or any other CUDA-capable GPU)
CUDA driver installed on the host, version ≥ 12.4
Docker with NVIDIA Container Toolkit (nvidia-docker)
WSL2 

## Setup instructions
1. Clone the repository.
2. Go to root/demo/sam2repo/sam2/checkpoints/ and run the script called download_ckpts.sh
3. Download Mediapipe's SelfieSegmenter model from: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
4. Download mediamtx rtsp server, start in and in another terminal window, run the following command: ffmpeg -f dshow -video_size 1280x720 -framerate 25 -i video="name of your webcam" -vcodec libx264 -preset ultrafast -tune zerolatency -vf "hflip" -f rtsp rtsp://localhost:8554/cam
5. Go to root/demo/ and enter: docker build -t "demo" . (This assumes you have Docker installed)
6. Enter .env values according to your setup, based on the env.example file
7. docker run --rm --gpus all --env-file .env -p 8000:8000 demo

## Usage
Once the app is running, open your browser and visit:

http://localhost:8000

You will see a live feed with the person segmented out and replaced with background pixels in real time.
