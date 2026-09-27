#!/usr/bin/env python3

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

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

# Now calculate right arm and wrist angles

def calculate_3d_angle(a, b, c):
    """Calculates the angle at point b given 3D coordinates a, b, c."""
    ab = a - b
    bc = c - b
    
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

angles = []
wrist_heights = []

# Group by frame to safely extract coordinates without pivot crashes
for frame_ind, group in df.groupby('frame_index'):
    # Map landmark IDs to their coordinate rows for this frame
    landmarks = {row['landmark_index']: row for _, row in group.iterrows()}
    
    # Ensure right shoulder(12), elbow(14), and wrist(16) are present
    if {12, 14, 16}.issubset(landmarks.keys()):
        s = landmarks[12]
        e = landmarks[14]
        w = landmarks[16]
        
        shoulder = np.array([s['x'], s['y'], s['z']])
        elbow = np.array([e['x'], e['y'], e['z']])
        wrist = np.array([w['x'], w['y'], w['z']])
        
        # Calculate elbow flexion angle
        angle = calculate_3d_angle(shoulder, elbow, wrist)
        angles.append({'frame': frame_ind, 'elbow_angle': angle})
        
        # Track vertical position (Y=0 is top of frame in MediaPipe)
        wrist_heights.append({
            'frame': frame_ind, 
            'wrist_y': w['y'], 
            'shoulder_y': s['y']
        })

metrics_df = pd.DataFrame(angles)

if len(metrics_df) > 0:
    peak_reach_frame = min(wrist_heights, key=lambda x: x['wrist_y'])['frame']
    print("--- Biomechanical Analysis Results ---")
    print(f"Successfully analyzed arm movement across {len(metrics_df)} frames.")
    print(f"Peak reach / Contact estimated at Frame: {peak_reach_frame}")
    print(metrics_df.describe())
else:
    print("Warning: No matching arm landmarks found across frames.")