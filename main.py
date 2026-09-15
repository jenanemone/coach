#!/usr/bin/env python3

import cv2

video_path = "IMG_2135.MOV"

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

cap.release()