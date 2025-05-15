import os
import torch
import numpy as np
import cv2
import subprocess

HOME = os.getcwd()

# Global variable for storing the selected point
selected_point = None

def click_event(event, x, y, flags, param):
    global selected_point, first_frame_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        # Mark the selected point on the image
        cv2.circle(first_frame_disp, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Point", first_frame_disp)

# Use torch.autocast in a context manager for cleaner usage
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    from sam2.build_sam import build_sam2_camera_predictor

    sam2_checkpoint = f"{HOME}/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    # Helper function: Overlay segmentation mask on the frame
    def overlay_mask(frame, mask, color=(255, 0, 0), alpha=0.6):
        """
        Overlays a colored mask on the frame.
        - frame: input image in BGR.
        - mask: binary mask (numpy array of shape (H, W)).
        - color: BGR tuple for the overlay color.
        - alpha: transparency factor.
        """
        mask_bool = np.squeeze(mask.astype(bool)) # Converting the mask into a boolean matrix
        overlay = np.zeros_like(frame, dtype=np.uint8) # Overlay with the masks' dimensions intially is black
        overlay[mask_bool] = color  # Applying the color to the mask
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0) # Blending the mask to the image 
        return blended

    # Helper function: Draw points on frame (if needed)
    def draw_points(frame, points, labels, marker_size=20):
        for pt, label in zip(points, labels):
            x, y = int(pt[0]), int(pt[1])
            if label == 1:
                cv2.drawMarker(frame, (x, y), (0, 255, 0),
                               markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2)
            else:
                cv2.drawMarker(frame, (x, y), (0, 0, 255),
                               markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2)
        return frame

    # Video capturing part
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

    # Read and process the first frame
    raw_frame = process.stdout.read(640 * 480 * 3)
    if not raw_frame:
        print("Error: Unable to read the first frame.")
        process.terminate()
        exit(1)

    frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))
    frame = cv2.resize(frame, (320, 240))  # Resize for processing/display
    # Load the first frame into the predictor (expects BGR)
    predictor.load_first_frame(frame)

    # Create a copy for user selection display
    first_frame_disp = frame.copy()
    cv2.imshow("Select Point", first_frame_disp)
    cv2.setMouseCallback("Select Point", click_event) # Selecting the prompted point

    # Wait until the user clicks a point (or press 'q' to quit)
    while selected_point is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            process.terminate()
            exit(0)

    # User has selected a point; close the selection window.
    cv2.destroyWindow("Select Point")
    
    # Use the selected point as the prompt (ensure it's in a numpy array of shape (1,2))
    points = np.array([selected_point], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # positive click
    ann_frame_idx = 0
    ann_obj_id = 1

    # Add the prompt to the predictor using the first frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Overlay initial segmentation mask on the first frame
    frame_disp = frame.copy()
    mask_initial = (out_mask_logits[0] > 0.0).cpu().numpy()
    frame_disp = overlay_mask(frame_disp, mask_initial)
    cv2.imshow("Segmented Video", frame_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        process.terminate()
        exit(0)

    # Process subsequent frames continuously
    while True:
        raw_frame = process.stdout.read(640 * 480 * 3)
        if not raw_frame:
            print("End of video stream or read error.")
            break

        # Convert raw frame to numpy array and resize
        frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))
        frame = cv2.resize(frame, (320, 240))
        
        if ann_frame_idx % 3 == 0:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            last_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            mask = last_mask
        
        else:
            mask = last_mask
        
        # Create a display copy and overlay the segmentation mask
        frame_disp = frame.copy()
        
        frame_disp = overlay_mask(frame_disp, mask)
        
        cv2.imshow("Segmented Video", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    process.terminate()
