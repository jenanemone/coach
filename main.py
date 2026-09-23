#!/usr/bin/env python3

import cv2
import mediapipe as mp
import pandas as pd

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

    frame_data = []

    print(f"Video properties: FPS={fps}, Frame Count={frame_count}, Duration={duration:.2f}s, Width={width}, Height={height}")

# Initialize MediaPipe Pose Estimator
with mp_pose.Pose(static_image_mode=False,model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:


    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = frame_index / fps
        
        
        # Perform pose estimation
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Loop through all 33 MediaPipe pose landmarks
            print(f"Successfully processed and tracked pose data across {frame_index} frames")
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_data.append({
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "landmark_index": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

            # Example: Grab the Right Wrist (landmark 16) and Right Shoulder (landmark 12)
            landmarks = results.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Print coordinates for the middle frame of your video as a sample
            if frame_index == 100:
                print(f"Frame 100 - Right Shoulder: (x={right_shoulder.x:.2f}, y={right_shoulder.y:.2f})")
                print(f"Frame 100 - Right Wrist: (x={right_wrist.x:.2f}, y={right_wrist.y:.2f})")

        else:
            print(f"No pose landmarks detected in frame {frame_index}")

        # Display the frame with pose estimation results (optional)
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1
# And... CUT!
cap.release()

# Now place into Pandas data frame 
df = pd.DataFrame(frame_data)
print(f"DataFrame successfully created with {len(df)} rows")
print(df.head())