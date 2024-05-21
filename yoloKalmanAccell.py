from ultralytics import YOLO
import cv2
import numpy as np

from KalmanFilter2D import KalmanFilter2D

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video path
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
time_between_frames = 1 / fps

# Initialize trackers
prev_positions = {}
prev_velocities = {}
kalman_filters = {}

# Fall detection thresholds
fall_threshold_x = 10.0  # Threshold for horizontal acceleration (m/s^2)
fall_threshold_y = 20.0  # Threshold for vertical acceleration (m/s^2)

def calculate_velocity(prev_pos, curr_pos, dt):
    return (curr_pos - prev_pos) / dt

def calculate_acceleration(prev_vel, curr_vel, dt):
    return (curr_vel - prev_vel) / dt

# Process video
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    # Get tracking results
    results = model.track(frame, persist=True)

    curr_positions = {}
    curr_velocities = {}

    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.id is not None:  # Ensure it's a person
                obj_id = int(box.id)
                center = (int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1] + box.xyxy[0][3]) / 2))
                curr_positions[obj_id] = np.array(center)

                if obj_id in prev_positions:
                    velocity = calculate_velocity(prev_positions[obj_id], curr_positions[obj_id], time_between_frames)
                    curr_velocities[obj_id] = velocity

                    if obj_id in prev_velocities:
                        acceleration = calculate_acceleration(prev_velocities[obj_id], velocity, time_between_frames)

                        # Initialize Kalman filter if not already
                        if obj_id not in kalman_filters:
                            A = np.eye(2)
                            H = np.eye(2)
                            Q = np.eye(2) * 0.01
                            R = np.eye(2) * 0.1
                            P = np.eye(2)
                            x0 = np.zeros(2)
                            kalman_filters[obj_id] = KalmanFilter2D(A, H, Q, R, P, x0)

                        # Kalman filter prediction and update
                        kf = kalman_filters[obj_id]
                        kf.predict()
                        kf.update(acceleration)
                        filtered_acceleration = kf.get_state()

                        # Fall detection logic
                        if abs(filtered_acceleration[0]) > fall_threshold_x or abs(filtered_acceleration[1]) > fall_threshold_y:
                            print("Fall detected!")
                            cv2.putText(frame, "Fall Detected!", (center[0], center[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Print or log the acceleration
                        print(f"Filtered Acceleration: ({filtered_acceleration[0]:.2f}, {filtered_acceleration[1]:.2f})")

                prev_positions[obj_id] = curr_positions[obj_id]
                prev_velocities[obj_id] = curr_velocities.get(obj_id, np.array([0, 0]))

    # Plot results on the frame
    frame_ = results[0].plot()

    # Display the frame
    cv2.imshow("frame", frame_)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
