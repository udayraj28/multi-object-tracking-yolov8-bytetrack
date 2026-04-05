# Technical Report: Multi-Object Detection and Persistent ID Tracking

## 1. Introduction
In this report, I'll walk through how I designed and implemented my multi-object tracking solution for the assignment. The goal was to take public sports footage, accurately detect the people in it, assign them stable IDs, and visualize their movement. I put together a pipeline using YOLOv8 for detection and ByteTrack for tracking, keeping the code as modular and clean as possible.

## 2. Detection Model (YOLOv8)
For the detection part, I decided to go with **YOLOv8m** (the medium variant). I tested a few options, but in my experience, YOLOv8 offers a really solid anchor-free architecture that predicts the center of objects well. In sports footage, players are often far away, so I couldn't use the smallest models because they drop detections too often. However, I also wanted the pipeline to run reasonably fast, so the medium size was the sweet spot. 

To keep the pipeline efficient and focused on the assignment requirements, I explicitly configured the model to filter out everything except class `0` (person in the MS-COCO dataset). 

## 3. Tracking Algorithm (ByteTrack)
To make sure the IDs stay the same across frames, I implemented **ByteTrack**. I've seen standard trackers struggle heavily because they simply throw away bounding boxes if the confidence score drops too low (which happens constantly during fast movement or partial occlusion). 

ByteTrack solves this nicely for me. It does a two-stage matching process:
1. It first links high-confidence boxes to existing tracks.
2. Instead of deleting the low-confidence boxes, it uses them to try and recover tracks that didn't get matched in step 1.

Since players in my video frequently pass behind each other, this ability to recycle lower-confidence detections saved me from getting constant ID switches.

## 4. Maintaining ID Consistency
To keep the IDs sticky, here is how the logic works under the hood:
*   I used the built-in Kalman Filters which estimate where a player is going to be in the next frame based on their current speed and direction.
*   When the new frame comes in, the algorithm calculates an Intersection over Union (IoU) between my Kalman prediction and the new bounding boxes YOLO found. 
*   I pass `persist=True` in my `detector.py` code, which tells the Ultralytics engine to maintain this state memory continuously across the loop.

## 5. Challenges Faced
*   **Visual Clutter:** At first, I started drawing the entire trajectory path for every player. Because it's a sports video, players move all over the place, and within a few seconds, the screen looked like a ball of yarn. I fixed this by capping my `TrajectoryManager` to only remember the last 30 coordinates. This created a nice "tail" effect instead.
*   **Type Conversions:** Handling the outputs from the YOLO tracker (which are PyTorch tensors) and converting them nicely into standard Integers and Numpy arrays so OpenCV could draw them required some careful parsing in the detection loop.

## 6. Failure Cases Observed
*   **Fast Panning:** When the camera operator suddenly pans to follow a fast play, the background shifts dramatically. Because ByteTrack relies heavily on spatial overlap (IoU) from frame to frame, a fast pan means the overlap drops to zero, and the tracker occasionally loses IDs.
*   **Pile-ups:** When multiple players get extremely close to each other, the bounding boxes merge entirely. The system sometimes gets confused about who is who once they separate, leading to ID swaps.

## 7. Possible Improvements
*   If I were taking this to the next level, I'd swap the tracker algorithm to something that uses **visual Re-Identification (like BoT-SORT)**. If the tracker knew the color of the jersey, it wouldn't get as confused during pile-ups.
*   I'd also implement an **Ego-Motion algorithm** to cancel out the camera panning before the Kalman filter makes its predictions. 

## 8. Why This Combination Was Selected
I ultimately chose the **YOLOv8 + ByteTrack** combination because it represents the best mix of accuracy, speed, and clean code. I didn't want to over-engineer the assignment with massive transformer models that take hours to set up or run at 1 frame per second. This approach allowed me to build a realistic, production-ready system that handles the stated problems well while adhering strictly to the time constraints of a 2-3 day assignment.
