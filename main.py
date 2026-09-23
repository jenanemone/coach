#!/usr/bin/env python3

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
video_path = "IMG_2135.MOV"

# 3,2,1... ACTION
print("coach is running!")

print(f"Attempting to open {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit(1)
else:
    print(f"Successfully opened video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: FPS={fps}, Frame Count={frame_count}, Duration={duration:.2f}s, Width={width}, Height={height}")

# Initialize MediaPipe Pose Estimator
with mp_pose.Pose(static_image_mode=False,model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1
        # Perform pose estimation
        results = pose.process(frame_rgb)

        if results.pose_landmarks:

            print(f"Successfully processed and tracked pose data across {frame_count} frames")

            # Example: Grab the Right Wrist (landmark 16) and Right Shoulder (landmark 12)
            landmarks = results.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Print coordinates for the middle frame of your video as a sample
            if frame_count == 100:
                print(f"Frame 100 - Right Shoulder: (x={right_shoulder.x:.2f}, y={right_shoulder.y:.2f})")
                print(f"Frame 100 - Right Wrist: (x={right_wrist.x:.2f}, y={right_wrist.y:.2f})")

        else:
            print(f"No pose landmarks detected in frame {frame_count}")

        # Display the frame with pose estimation results (optional)
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# And... CUT!
cap.release()