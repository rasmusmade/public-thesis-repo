import cv2
import numpy as np
import subprocess
import time
from screeninfo import get_monitors
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tf_pose_estimation'))

from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

# === Model Setup ===
model = 'mobilenet_thin'  # Fast and light model
resize = '432x368'        # Input resolution for the model (adjustable)

w, h = model_wh(resize)
pose_estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# === Display Setup ===
monitor = get_monitors()[0]
SCREEN_WIDTH = monitor.width
SCREEN_HEIGHT = monitor.height

cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation", SCREEN_WIDTH, SCREEN_HEIGHT)

# === RTSP Stream Setup ===
rtspURL = "rtsp://172.17.154.156:8554/mystream?rtsp_transport=tcp"
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

# === Runtime Info ===
frame_counter = 0
start_time = time.time()

# === Frame Display Function ===
def resize_and_show(image, fps):
    if image is None or image.size == 0:
        return
    resized = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    cv2.putText(resized, f"{fps:.1f} FPS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("Pose Estimation", resized)

# === Main Loop ===
try:
    while True:
        raw_frame = process.stdout.read(1280 * 720 * 3)
        if len(raw_frame) != 1280 * 720 * 3:
            continue

        try:
            frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))
        except ValueError:
            continue

        frame_counter += 1
        inf_start = time.time()

        humans = pose_estimator.inference(frame, resize_to_default=True, upsample_size=4.0)
        output_frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        inf_end = time.time()
        elapsed = time.time() - start_time
        fps = frame_counter / elapsed if elapsed > 0 else 0

        resize_and_show(output_frame, fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    process.terminate()
    cv2.destroyAllWindows()
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_fps = frame_counter / total_runtime if total_runtime else 0
    print(f"Total frames: {frame_counter}")
    print(f"Runtime: {total_runtime:.2f}s")
    print(f"Avg FPS: {avg_fps:.2f}")
