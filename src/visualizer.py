import cv2
import numpy as np
from typing import List, Tuple
from src.trajectory import TrajectoryManager

class Visualizer:
    """
    Handles drawing bounding boxes, labels, and trajectories on video frames.
    """
    def __init__(self):
        # Optional: pre-generate colors or use a simple hashing technique based on track_id
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, (1000, 3)).tolist()

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Gets a consistent color for a given track ID."""
        color = self.colors[track_id % len(self.colors)]
        return tuple(color)

    def draw_annotations(self, frame: np.ndarray, boxes: List[List[float]], track_ids: List[int], trajectory_manager: TrajectoryManager) -> np.ndarray:
        """
        Annotates the frame explicitly with bounding boxes, IDs, and trajectory tails.
        
        Args:
            frame: the image frame
            boxes: bounding boxes [x1, y1, x2, y2]
            track_ids: list of integer IDs
            trajectory_manager: TrajectoryManager instance holding the history
            
        Returns:
            Annotated frame.
        """
        annotated_frame = frame.copy()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            color = self.get_color(track_id)
            
            # Draw Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"ID: {track_id}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw Trajectory
            # Retrieve history
            history = trajectory_manager.get_history(track_id)
            if len(history) > 1:
                # Convert list of tuples to numpy array shaped exactly like cv2.polylines expects
                pts = np.array(history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

        return annotated_frame
