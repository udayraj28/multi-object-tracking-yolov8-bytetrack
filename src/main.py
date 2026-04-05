import argparse
import sys
import cv2

from src.utils import get_video_properties, create_video_writer, setup_logger
from src.detector import DetectorTracker
from src.trajectory import TrajectoryManager
from src.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Object Tracking with YOLOv8 and ByteTrack")
    parser.add_argument("--input_video", type=str, default="input/sample_video.mp4", help="Path to the input video file")
    parser.add_argument("--output_video", type=str, default="output/tracked_output.mp4", help="Path to save the output video file")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("Main")
    
    logger.info(f"Input video: {args.input_video}")
    logger.info(f"Output video: {args.output_video}")
    
    # 1. Initialize Video Reader and Writer
    cap, fps, width, height = get_video_properties(args.input_video)
    writer = create_video_writer(args.output_video, fps, width, height)
    
    # 2. Initialize Core Components
    detector_tracker = DetectorTracker(model_name="yolov8m.pt")
    trajectory_manager = TrajectoryManager(max_history=30)
    visualizer = Visualizer()
    
    frame_count = 0
    logger.info("Starting video processing pipeline...")
    
    try:
        # 3. Read Frames Sequentially
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break
                
            # 4. Run Detection + Tracking
            bboxes, confidences, track_ids = detector_tracker.process_frame(frame)
            
            # 5. Update Trajectories
            for box, track_id in zip(bboxes, track_ids):
                # Calculate center of the bounding box
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                trajectory_manager.update(track_id, center_x, center_y)
                
            # 6. Draw Bounding Boxes, Labels, and Trajectories
            annotated_frame = visualizer.draw_annotations(frame, bboxes, track_ids, trajectory_manager)
            
            # 7. Save Annotated Frame
            writer.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames...")
                
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        
    finally:
        # 8. Release Resources
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Processing complete. Output saved to {args.output_video}")

if __name__ == "__main__":
    main()
