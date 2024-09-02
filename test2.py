import re
import numpy as np
import cv2
from pathlib import Path

threshold = 5
aspect_ratio1 = 0.2
aspect_ratio2 = 0.8

def pre_crop(frame, crop_x, crop_y, crop_w, crop_h):
    return frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

VIDEO_PATH = 'datasets/fenix/fall-01-cam0.mp4'
# x, y, w, h
CROP_REGION_RELATIVE = (0.5, 0, 0.5, 1)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Cropping
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_x, crop_y, crop_w, crop_h = [
    int(v)
    for v in (
        orig_w * CROP_REGION_RELATIVE[0],
        orig_h * CROP_REGION_RELATIVE[1],
        orig_w * CROP_REGION_RELATIVE[2],
        orig_h * CROP_REGION_RELATIVE[3],
    )
]

frames: list[np.ndarray] = []

# Define the regex patterns for the directories
pattern1 = r'datasets/fenix/.*'
pattern2 = r'fall_datasets/.*'

# Check if the path matches the 'datasets/fenix/' pattern
is_in_datasets_fenix = re.match(pattern1, VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if is_in_datasets_fenix:
        frame = pre_crop(frame, crop_x, crop_y, crop_w, crop_h)
        frames.append(frame)
    else:
        frames.append(frame)

# Background estimation
bg = np.median(frames, axis=0).astype(np.uint8)
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# Subtract frame from background
frames_mask = []
for frame in frames:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame_gray, bg_gray)
    mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    frames_mask.append(mask)

# Kalman filter initialization
kalman = cv2.KalmanFilter(8, 4)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03

# Initialize variables for fall detection
previous_center = None
fall_detected = False
aspect_ratios = []

# Process each frame
for frame, mask in zip(frames, frames_mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)  # Yellow color for contours

    # Filter out small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    if contours:
        # Combine all contours into one
        all_contours = np.vstack(contours)

        # Get the bounding box of the combined contours
        x, y, w, h = cv2.boundingRect(all_contours)

        # Calculate the center of the bounding box
        cx, cy = x + w // 2, y + h // 2

        # Measurement update (bounding box center and size)
        measurement = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(w)], [np.float32(h)]])
        kalman.correct(measurement)

        # Predict next position and size
        prediction = kalman.predict()
        pred_cx, pred_cy, pred_w, pred_h = int(prediction[0]), int(prediction[1]), int(prediction[4]), int(prediction[5])

        # Draw the predicted bounding box
        pred_x, pred_y = pred_cx - pred_w // 2, pred_cy - pred_h // 2
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 255, 0), 2)
        cv2.circle(frame, (pred_cx, pred_cy), 5, (0, 0, 255), -1)  # Draw the center dot

        # Check for fall
        if previous_center is not None:
            # Calculate velocity
            velocity = np.linalg.norm(np.array([cx - previous_center[0], cy - previous_center[1]]))

            # Calculate aspect ratio
            aspect_ratio = h / w if w != 0 else 0
            aspect_ratios.append(aspect_ratio)

            # Detect fall based on velocity and aspect ratio change
            if velocity > threshold and len(aspect_ratios) > 1:  # Adjust the threshold as needed
                if aspect_ratios[-2] > aspect_ratio1 and aspect_ratios[-1] < aspect_ratio2:  # Adjust the ratios as needed
                    fall_detected = True

        previous_center = (cx, cy)
    else:
        # No object detected, predict next position and size
        prediction = kalman.predict()
        pred_cx, pred_cy, pred_w, pred_h = int(prediction[0]), int(prediction[1]), int(prediction[4]), int(prediction[5])

        # Draw the predicted bounding box
        pred_x, pred_y = pred_cx - pred_w // 2, pred_cy - pred_h // 2
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (255, 0, 0), 2)
        cv2.circle(frame, (pred_cx, pred_cy), 5, (255, 0, 0), -1)  # Draw the predicted center dot

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if fall_detected:
        cv2.putText(frame_rgb, "Fall Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
