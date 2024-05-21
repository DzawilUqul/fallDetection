import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Curl counter variables
counter = 0
counter_two = 0
counter_three = 0
counter_four = 0
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image_height, image_width, _ = image.shape

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # ----------------------   DOT   ----------------------           

            # dot - NOSE

            dot_NOSE_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)
            dot_NOSE_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height)

            # dot - LEFT_SHOULDER

            dot_LEFT_SHOULDER_X = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
            dot_LEFT_SHOULDER_Y = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)

            # dot - RIGHT_SHOULDER

            dot_RIGHT_SHOULDER_X = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
            dot_RIGHT_SHOULDER_Y = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)

            # dot - LEFT_ELBOW

            dot_LEFT_ELBOW_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width)
            dot_LEFT_ELBOW_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height)

            # dot - RIGHT_ELBOW

            dot_RIGHT_ELBOW_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width)
            dot_RIGHT_ELBOW_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height)

            # dot - LEFT_WRIST

            dot_LEFT_WRIST_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
            dot_LEFT_WRIST_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)

            # dot - RIGHT_WRIST

            dot_RIGHT_WRIST_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
            dot_RIGHT_WRIST_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)

            # 2작업

            # dot - LEFT_HIP

            dot_LEFT_HIP_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
            dot_LEFT_HIP_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)

            # dot - RIGHT_HIP

            dot_RIGHT_HIP_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
            dot_RIGHT_HIP_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)

            # dot - LEFT_KNEE

            dot_LEFT_KNEE_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
            dot_LEFT_KNEE_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)

            # dot - RIGHT_KNEE

            dot_RIGHT_KNEE_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)
            dot_RIGHT_KNEE_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)

            # dot - LEFT_ANKLE

            dot_LEFT_ANKLE_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width)
            dot_LEFT_ANKLE_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height)

            # dot - RIGHT_ANKLE

            dot_RIGHT_ANKLE_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width)
            dot_RIGHT_ANKLE_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height)

            # dot - LEFT_HEEL

            dot_LEFT_HEEL_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * image_width)
            dot_LEFT_HEEL_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * image_height)

            # dot - RIGHT_HEEL

            dot_RIGHT_HEEL_X = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width)
            dot_RIGHT_HEEL_Y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * image_height)

            # dot - LEFT_FOOT_INDEX

            dot_LEFT_FOOT_INDEX_X = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width)
            dot_LEFT_FOOT_INDEX_Y = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_height)

            # dot - LRIGHTFOOT_INDEX

            dot_RIGHT_FOOT_INDEX_X = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width)
            dot_RIGHT_FOOT_INDEX_Y = int(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_height)

            # dot - NOSE

            dot_NOSE = [dot_NOSE_X, dot_NOSE_Y]

            # dot - LEFT_ARM_WRIST_ELBOW

            dot_LEFT_ARM_A_X = int((dot_LEFT_WRIST_X + dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_A_Y = int((dot_LEFT_WRIST_Y + dot_LEFT_ELBOW_Y) / 2)

            LEFT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X, dot_LEFT_ARM_A_Y]

            # dot - RIGHT_ARM_WRIST_ELBOW

            dot_RIGHT_ARM_A_X = int((dot_RIGHT_WRIST_X + dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_A_Y = int((dot_RIGHT_WRIST_Y + dot_RIGHT_ELBOW_Y) / 2)

            RIGHT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X, dot_LEFT_ARM_A_Y]

            # dot - LEFT_ARM_SHOULDER_ELBOW

            dot_LEFT_ARM_SHOULDER_ELBOW_X = int((dot_LEFT_SHOULDER_X + dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_SHOULDER_ELBOW_Y = int((dot_LEFT_SHOULDER_Y + dot_LEFT_ELBOW_Y) / 2)

            LEFT_ARM_SHOULDER_ELBOW = [dot_LEFT_ARM_SHOULDER_ELBOW_X, dot_LEFT_ARM_SHOULDER_ELBOW_Y]

            # dot - RIGHT_ARM_SHOULDER_ELBOW

            dot_RIGHT_ARM_SHOULDER_ELBOW_X = int((dot_RIGHT_SHOULDER_X + dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_SHOULDER_ELBOW_Y = int((dot_RIGHT_SHOULDER_Y + dot_RIGHT_ELBOW_Y) / 2)

            RIGHT_ARM_SHOULDER_ELBOW = [dot_RIGHT_ARM_SHOULDER_ELBOW_X, dot_RIGHT_ARM_SHOULDER_ELBOW_Y]

            # dot - BODY_SHOULDER_HIP

            dot_BODY_SHOULDER_HIP_X = int(
                (dot_RIGHT_SHOULDER_X + dot_RIGHT_HIP_X + dot_LEFT_SHOULDER_X + dot_LEFT_HIP_X) / 4)
            dot_BODY_SHOULDER_HIP_Y = int(
                (dot_RIGHT_SHOULDER_Y + dot_RIGHT_HIP_Y + dot_LEFT_SHOULDER_Y + dot_LEFT_HIP_Y) / 4)

            BODY_SHOULDER_HIP = [dot_BODY_SHOULDER_HIP_X, dot_BODY_SHOULDER_HIP_Y]

            # dot - LEFT_LEG_HIP_KNEE

            dot_LEFT_LEG_HIP_KNEE_X = int((dot_LEFT_HIP_X + dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_HIP_KNEE_Y = int((dot_LEFT_HIP_Y + dot_LEFT_KNEE_Y) / 2)

            LEFT_LEG_HIP_KNEE = [dot_LEFT_LEG_HIP_KNEE_X, dot_LEFT_LEG_HIP_KNEE_Y]

            # dot - RIGHT_LEG_HIP_KNEE

            dot_RIGHT_LEG_HIP_KNEE_X = int((dot_RIGHT_HIP_X + dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_HIP_KNEE_Y = int((dot_RIGHT_HIP_Y + dot_RIGHT_KNEE_Y) / 2)

            RIGHT_LEG_HIP_KNEE = [dot_RIGHT_LEG_HIP_KNEE_X, dot_RIGHT_LEG_HIP_KNEE_Y]

            # dot - LEFT_LEG_KNEE_ANKLE

            dot_LEFT_LEG_KNEE_ANKLE_X = int((dot_LEFT_ANKLE_X + dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_KNEE_ANKLE_Y = int((dot_LEFT_ANKLE_Y + dot_LEFT_KNEE_Y) / 2)

            LEFT_LEG_KNEE_ANKLE = [dot_LEFT_LEG_KNEE_ANKLE_X, dot_LEFT_LEG_KNEE_ANKLE_Y]

            # dot - RIGHT_LEG_KNEE_ANKLE

            dot_RIGHT_LEG_KNEE_ANKLE_X = int((dot_RIGHT_ANKLE_X + dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_KNEE_ANKLE_Y = int((dot_RIGHT_ANKLE_Y + dot_RIGHT_KNEE_Y) / 2)

            RIGHT_LEG_KNEE_ANKLE = [dot_RIGHT_LEG_KNEE_ANKLE_X, dot_RIGHT_LEG_KNEE_ANKLE_Y]

            # dot - LEFT_FOOT_INDEX_HEEL

            dot_LEFT_FOOT_INDEX_HEEL_X = int((dot_LEFT_FOOT_INDEX_X + dot_LEFT_HEEL_X) / 2)
            dot_LEFT_FOOT_INDEX_HEEL_Y = int((dot_LEFT_FOOT_INDEX_Y + dot_LEFT_HEEL_Y) / 2)

            LEFT_FOOT_INDEX_HEEL = [dot_LEFT_FOOT_INDEX_HEEL_X, dot_LEFT_FOOT_INDEX_HEEL_Y]

            # dot - RIGHT_FOOT_INDEX_HEEL

            dot_RIGHT_FOOT_INDEX_HEEL_X = int((dot_RIGHT_FOOT_INDEX_X + dot_RIGHT_HEEL_X) / 2)
            dot_RIGHT_FOOT_INDEX_HEEL_Y = int((dot_RIGHT_FOOT_INDEX_Y + dot_RIGHT_HEEL_Y) / 2)

            RIGHT_FOOT_INDEX_HEEL = [dot_RIGHT_FOOT_INDEX_HEEL_X, dot_RIGHT_FOOT_INDEX_HEEL_Y]

            # dot _ UPPER_BODY

            dot_UPPER_BODY_X = int((
                                           dot_NOSE_X + dot_LEFT_ARM_A_X + dot_RIGHT_ARM_A_X + dot_LEFT_ARM_SHOULDER_ELBOW_X + dot_RIGHT_ARM_SHOULDER_ELBOW_X + dot_BODY_SHOULDER_HIP_X) / 6)
            dot_UPPER_BODY_Y = int((
                                           dot_NOSE_Y + dot_LEFT_ARM_A_Y + dot_RIGHT_ARM_A_Y + dot_LEFT_ARM_SHOULDER_ELBOW_Y + dot_RIGHT_ARM_SHOULDER_ELBOW_Y + dot_BODY_SHOULDER_HIP_Y) / 6)

            UPPER_BODY = [dot_UPPER_BODY_X, dot_UPPER_BODY_Y]

            # dot _ LOWER_BODY

            dot_LOWER_BODY_X = int((
                                           dot_LEFT_LEG_HIP_KNEE_X + dot_RIGHT_LEG_HIP_KNEE_X + dot_LEFT_LEG_KNEE_ANKLE_X + dot_RIGHT_LEG_KNEE_ANKLE_X + dot_LEFT_FOOT_INDEX_HEEL_X + dot_RIGHT_FOOT_INDEX_HEEL_X) / 6)
            dot_LOWER_BODY_Y = int((
                                           dot_LEFT_LEG_HIP_KNEE_Y + dot_RIGHT_LEG_HIP_KNEE_Y + dot_LEFT_LEG_KNEE_ANKLE_Y + dot_RIGHT_LEG_KNEE_ANKLE_Y + dot_LEFT_FOOT_INDEX_HEEL_Y + dot_RIGHT_FOOT_INDEX_HEEL_Y) / 6)

            LOWER_BODY = [dot_LOWER_BODY_X, dot_LOWER_BODY_Y]

            # dot _ BODY

            dot_BODY_X = int((dot_UPPER_BODY_X + dot_LOWER_BODY_X) / 2)
            dot_BODY_Y = int((dot_UPPER_BODY_Y + dot_LOWER_BODY_Y) / 2)

            BODY = [dot_BODY_X, dot_BODY_Y]

            # ---------------------------  COOLDINATE  ---------------------- 

            # Get coordinates - elbow_l
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates - elbow_r
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Get coordinates - shoulder_l
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Get coordinates - shoulder_r
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Get coordinates - hip_l
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Get coordinates - hip_r
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Get coordinates - knee_l
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Get coordinates - knee_r
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle - elbow_l
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

            # Calculate angle - elbow_r
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

            # Calculate angle - shoulder_l
            angle_shoulder_l = calculate_angle(elbow_l, shoulder_l, hip_l)

            # Calculate angle - shoulder_r
            angle_shoulder_r = calculate_angle(elbow_r, shoulder_r, hip_r)

            # Calculate angle - hip_l
            angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)

            # Calculate angle - hip_r
            angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)

            # Calculate angle - knee_l
            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)

            # Calculate angle - knee_r
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)

            # 발 사이값
            Point_of_action_LEFT_X = int(
                ((dot_LEFT_FOOT_INDEX_X + dot_LEFT_HEEL_X) / 2))

            Point_of_action_LEFT_Y = int(
                ((dot_LEFT_FOOT_INDEX_Y + dot_LEFT_HEEL_Y) / 2))

            Point_of_action_RIGHT_X = int(
                ((dot_RIGHT_FOOT_INDEX_X + dot_RIGHT_HEEL_X) / 2))

            Point_of_action_RIGHT_Y = int(
                ((dot_RIGHT_FOOT_INDEX_Y + dot_RIGHT_HEEL_Y) / 2))

            # 발 사이값 평균

            Point_of_action_X = int((Point_of_action_LEFT_X + Point_of_action_RIGHT_X) / 2)

            Point_of_action_Y = int((Point_of_action_LEFT_Y + Point_of_action_RIGHT_Y) / 2)

            # 발 사이값 좌표
            Point_of_action = [Point_of_action_X, Point_of_action_Y]

            # Visualize angle - 발 사이값 좌표

            # cv2.putText(image, str(Point_of_action),
            #             (Point_of_action_X, Point_of_action_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (Point_of_action_X, Point_of_action_Y), 5, (0, 0, 255), -1)

            # # Visualize angle - elbow_l
            # cv2.putText(image, str(angle_elbow_l),
            #             tuple(np.multiply(elbow_l, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - elbow_r
            # cv2.putText(image, str(angle_elbow_r),
            #             tuple(np.multiply(elbow_r, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - shoulder_l
            # cv2.putText(image, str(angle_shoulder_l),
            #             tuple(np.multiply(shoulder_l, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - shoulder_r
            # cv2.putText(image, str(angle_shoulder_r),
            #             tuple(np.multiply(shoulder_r, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - hip_l
            # cv2.putText(image, str(angle_hip_l),
            #             tuple(np.multiply(hip_l, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - hip_r
            # cv2.putText(image, str(angle_hip_r),
            #             tuple(np.multiply(hip_r, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - knee_l
            # cv2.putText(image, str(angle_knee_l),
            #             tuple(np.multiply(knee_l, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize angle - knee_r
            # cv2.putText(image, str(angle_knee_r),
            #             tuple(np.multiply(knee_r, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            #             )
            # 
            # # Visualize dot - dot_NOSE
            # 
            # cv2.putText(image, str(dot_NOSE),
            #             (dot_NOSE_X, dot_NOSE_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_NOSE_X, dot_NOSE_Y), 5, (204, 252, 0), -1)

            # Visualize dot - LEFT_ARM_WRIST_ELBO

            # cv2.putText(image, str(LEFT_ARM_WRIST_ELBOW),
            #             (dot_LEFT_ARM_A_X, dot_LEFT_ARM_A_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )
            # 
            # cv2.circle(image, (dot_LEFT_ARM_A_X, dot_LEFT_ARM_A_Y), 5, (204, 252, 0), -1)
            # 
            # # Visualize dot - RIGHT_ARM_WRIST_ELBO
            # 
            # cv2.putText(image, str(RIGHT_ARM_WRIST_ELBOW),
            #             (dot_RIGHT_ARM_A_X, dot_RIGHT_ARM_A_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_RIGHT_ARM_A_X, dot_RIGHT_ARM_A_Y), 5, (204, 252, 0), -1)

            # Visualize dot - LEFT_ARM_SHOULDER_ELBOW

            # cv2.putText(image, str(LEFT_ARM_SHOULDER_ELBOW),
            #             (dot_LEFT_ARM_SHOULDER_ELBOW_X, dot_LEFT_ARM_SHOULDER_ELBOW_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_LEFT_ARM_SHOULDER_ELBOW_X, dot_LEFT_ARM_SHOULDER_ELBOW_Y), 5, (204, 252, 0), -1)

            # Visualize dot - RIGHT_ARM_SHOULDER_ELBOW

            # cv2.putText(image, str(RIGHT_ARM_SHOULDER_ELBOW),
            #             (dot_RIGHT_ARM_SHOULDER_ELBOW_X, dot_RIGHT_ARM_SHOULDER_ELBOW_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_RIGHT_ARM_SHOULDER_ELBOW_X, dot_RIGHT_ARM_SHOULDER_ELBOW_Y), 5, (204, 252, 0), -1)

            # Visualize dot - BODY_SHOULDER_HIP

            # cv2.putText(image, str(BODY_SHOULDER_HIP),
            #             (dot_BODY_SHOULDER_HIP_X, dot_BODY_SHOULDER_HIP_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_BODY_SHOULDER_HIP_X, dot_BODY_SHOULDER_HIP_Y), 5, (204, 252, 0), -1)

            # Visualize dot - LEFT_LEG_HIP_KNEE

            # cv2.putText(image, str(LEFT_LEG_HIP_KNEE),
            #             (dot_LEFT_LEG_HIP_KNEE_X, dot_LEFT_LEG_HIP_KNEE_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_LEFT_LEG_HIP_KNEE_X, dot_LEFT_LEG_HIP_KNEE_Y), 5, (204, 252, 0), -1)

            # Visualize dot - RIGHT_LEG_HIP_KNEE

            # cv2.putText(image, str(RIGHT_LEG_HIP_KNEE),
            #             (dot_RIGHT_LEG_HIP_KNEE_X, dot_RIGHT_LEG_HIP_KNEE_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_RIGHT_LEG_HIP_KNEE_X, dot_RIGHT_LEG_HIP_KNEE_Y), 5, (204, 252, 0), -1)

            # Visualize dot - LEFT_LEG_KNEE_ANKLE

            # cv2.putText(image, str(LEFT_LEG_KNEE_ANKLE),
            #             (dot_LEFT_LEG_KNEE_ANKLE_X, dot_LEFT_LEG_KNEE_ANKLE_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_LEFT_LEG_KNEE_ANKLE_X, dot_LEFT_LEG_KNEE_ANKLE_Y), 5, (204, 252, 0), -1)

            # Visualize dot - RIGHT_LEG_KNEE_ANKLE

            # cv2.putText(image, str(RIGHT_LEG_KNEE_ANKLE),
            #             (dot_RIGHT_LEG_KNEE_ANKLE_X, dot_RIGHT_LEG_KNEE_ANKLE_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_RIGHT_LEG_KNEE_ANKLE_X, dot_RIGHT_LEG_KNEE_ANKLE_Y), 5, (204, 252, 0), -1)

            # Visualize dot -   LEFT_FOOT_INDEX_HEEL

            # cv2.putText(image, str(LEFT_FOOT_INDEX_HEEL),
            #             (dot_LEFT_FOOT_INDEX_HEEL_X, dot_LEFT_FOOT_INDEX_HEEL_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_LEFT_FOOT_INDEX_HEEL_X, dot_LEFT_FOOT_INDEX_HEEL_Y), 5, (204, 252, 0), -1)

            # Visualize dot -   RIGHT_FOOT_INDEX_HEEL

            # cv2.putText(image, str(RIGHT_FOOT_INDEX_HEEL),
            #             (dot_RIGHT_FOOT_INDEX_HEEL_X, dot_RIGHT_FOOT_INDEX_HEEL_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 252, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_RIGHT_FOOT_INDEX_HEEL_X, dot_RIGHT_FOOT_INDEX_HEEL_Y), 5, (204, 252, 0), -1)

            # Visualize dot -   UPPER_BODY

            # cv2.putText(image, str(UPPER_BODY),
            #             (dot_UPPER_BODY_X, dot_UPPER_BODY_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277, 220, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_UPPER_BODY_X, dot_UPPER_BODY_Y), 9, (277, 220, 0), -1)

            # Visualize dot -   LOWER_BODY

            # cv2.putText(image, str(LOWER_BODY),
            #             (dot_LOWER_BODY_X, dot_LOWER_BODY_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277, 220, 0), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_LOWER_BODY_X, dot_LOWER_BODY_Y), 9, (277, 220, 0), -1)

            # Visualize dot -   BODY

            # cv2.putText(image, str(BODY),
            #             (dot_BODY_X, dot_BODY_Y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            #             )

            cv2.circle(image, (dot_BODY_X, dot_BODY_Y), 12, (0, 0, 255), -1)

            # fall case
            fall = int(Point_of_action_X - dot_BODY_X)

            # --------------------------   여기까지                     
            # case falling and standa

            falling = abs(fall) > 50
            standing = abs(fall) < 50

            x = Point_of_action_X
            y = -(1.251396648 * x) + 618

            if falling:
                stage = "falling"
            #   if Point_of_action_X <  320 and Point_of_action_Y > 240 and standing and stage == 'falling':     #count3            
            #       cv2.putText(image, 'fall' , ( 320,240 ),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA )
            #       stage = "standing"
            #       counter_three +=1
            if Point_of_action_X < 320 and Point_of_action_X > 100 and Point_of_action_Y > 390 and Point_of_action_Y > y and standing and stage == 'falling':  # count3            
                cv2.putText(image, 'Fall Detected', (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                stage = "standing"
                counter_three += 1
                print(Point_of_action, y)
            if Point_of_action_X >= 320 and Point_of_action_Y > 240 and standing and stage == 'falling':  # count4                
                cv2.putText(image, 'fall', (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                stage = "standing"
                counter_four += 1


        except:
            pass
        # -------------------------------

        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, "Fall Detected", (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, str(counter + counter_two + counter_three + counter_four),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'distance', (150, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # # count 1
        # cv2.putText(image, str(counter),
        #             (195, 305),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (281, 305),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (358, 305),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (454, 305),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # 
        # cv2.putText(image, str(counter),
        #             (138, 363),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (272, 363),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (380, 363),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # 
        # cv2.putText(image, str(counter),
        #             (43, 433),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # 
        # cv2.putText(image, str(counter),
        #             (397, 433),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # cv2.putText(image, str(counter),
        #             (578, 433),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)  # count 1
        # 
        # # count 3
        # cv2.putText(image, str(counter_three),
        #             (246, 428),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # 
        # # count 4
        # cv2.putText(image, str(counter_four),
        #             (516, 350),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 100, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
