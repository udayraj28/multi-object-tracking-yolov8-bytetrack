# Multi-Object Detection and Persistent ID Tracking

**Source Video Link:** [https://youtube.com/shorts/kAVCAP5D3cs?si=0C-9DbzGSU8vwqqj](https://youtube.com/shorts/kAVCAP5D3cs?si=0C-9DbzGSU8vwqqj)

## 1. Project Overview
For this assignment, I built a computer vision pipeline to detect multiple people and track them with persistent IDs in sports footage. I wanted to make sure the system was robust, so I focused on correctly identifying individuals, assigning them an ID, and drawing out their movement paths over time. I structured the code to be modular so it's easy to read and scale.

## 2. Architecture Diagram (ASCII)

```text
+-----------------------+      +-----------------------+      +-----------------------+
|                       |      |                       |      |                       |
|  Input Video Stream   +----->|  DetectorTracker      +----->|  TrajectoryManager    |
|  (Frames Extracted)   |      |  (YOLOv8 + ByteTrack) |      |  (x,y coords & IDs)   |
+-----------------------+      +----------+------------+      +----------+------------+
                                          |                              |
                                          v                              v
                               +------------------------------------------------------+
                               |                                                      |
                               |                   Visualizer                         |
                               |    (Draws Bounding Boxes, IDs, & Polylines)          |
                               +--------------------------+---------------------------+
                                                          |
                                                          |
                                                          v
                               +------------------------------------------------------+
                               |                                                      |
                               |                  Output Video                        |
                               |               (Annotated Frames)                     |
                               +------------------------------------------------------+
```

## 3. Installation Steps

1. Clone or download this repository.
2. I recommend creating a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 4. How to Run

Before running, place your input video at `input/sample_video.mp4` (or just pass the custom path in the command below).

To start the tracking pipeline, run:

```bash
python -m src.main --input_video input/sample_video.mp4 --output_video output/tracked_output.mp4
```

On the first run, the script will download the `yolov8m.pt` weights automatically. You'll see frame processing logs in your console, and the final annotated video will show up in the `output/` folder.

## 5. Model and Tracker Choices

*   **Model**: I chose the **YOLOv8m** (medium) model. It gives a really good balance between speed and accuracy. In sports footage, players can be small or moving fast, so the 'nano' or 'small' models miss too much, but the 'large' model is overkill and too slow. I explicitly filtered the detections for class `0` (person) so the tracker doesn't get confused by the ball or other objects.
*   **Tracker**: I used **ByteTrack** for assigning IDs. What I like about ByteTrack is that it recovers IDs efficiently even when player detection scores temporarily drop due to occlusion.

## 6. Assumptions
*   I assumed we only care about tracking humans (players, referees, etc.), so I ignored all other classes.
*   I assumed the video has decent lighting and resolution.
*   For the trajectory lines, I capped the history at 30 frames. I made this assumption because drawing the entire history makes sports videos way too cluttered to look at.

## 7. Limitations
*   **Long-Term Occlusion:** If a player gets fully blocked from the camera for too long, ByteTrack might lose track of them and assign a new ID when they reappear.
*   **Speed vs Hardware:** I didn't optimize this for pure CPU inference, so running `yolov8m` without a GPU might be a bit slow, though it still works perfectly fine.
