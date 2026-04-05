import numpy as np
from typing import List, Tuple, Any
from ultralytics import YOLO
from src.utils import setup_logger

class DetectorTracker:
    """
    Combined Detector and Tracker using YOLOv8 natively supported ByteTrack.
    This fulfills both detection (YOLOv8m) and tracking (ByteTrack) requirements.
    Detects "person" class only.
    """
    def __init__(self, model_name: str = "yolov8m.pt"):
        self.logger = setup_logger("DetectorTracker")
        self.logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        # Class 0 corresponds to 'person' in COCO dataset
        self.target_class_id = 0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Runs tracking on a frame using ByteTrack.
        
        Args:
            frame: The image frame.
            
        Returns:
            Tuple containing:
            - bboxes: List of bounding boxes [x1, y1, x2, y2].
            - confidences: List of confidences.
            - track_ids: List of unique integer track IDs.
        """
        # Run YOLO tracking with ByteTrack, persist=True ensures IDs are persistent across frames
        # classes=[0] filters for 'person'
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[self.target_class_id], 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        bboxes = []
        confidences = []
        track_ids = []
        
        # Parse the results
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                bboxes = boxes.xyxy.cpu().numpy().tolist()
                confidences = boxes.conf.cpu().numpy().tolist()
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
                
        return bboxes, confidences, track_ids
