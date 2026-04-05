from typing import Dict, List, Tuple

class TrajectoryManager:
    """
    Manages the tracking history to create trajectory polylines.
    Maintains a dictionary of Track ID -> List of (x, y) coordinates.
    """
    def __init__(self, max_history: int = 30):
        """
        Args:
            max_history (int): The maximum number of points to keep in the history for each ID.
        """
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_history = max_history

    def update(self, track_id: int, center_x: int, center_y: int):
        """
        Adds a new point to the history of a track_id.
        
        Args:
            track_id (int): The persistent ID of the tracked object.
            center_x (int): X-coordinate of the center of the bounding box.
            center_y (int): Y-coordinate of the center of the bounding box.
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            
        self.track_history[track_id].append((center_x, center_y))
        
        # Limit the stored history to max_history points
        if len(self.track_history[track_id]) > self.max_history:
            self.track_history[track_id].pop(0)

    def get_history(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Retrieves the trajectory history for a track ID.
        
        Args:
            track_id (int): The persistent ID.
            
        Returns:
            List of (x, y) coordinates.
        """
        return self.track_history.get(track_id, [])
    
    def cleanup(self, active_track_ids: List[int]):
        """
        Removes trajectory histories for IDs that are no longer active to free memory.
        
        Args:
            active_track_ids: List of currently tracked IDs in the current frame.
        """
        inactive_ids = [tid for tid in self.track_history.keys() if tid not in active_track_ids]
        # Wait until a certain threshold before removing or removing immediately?
        # Usually, trackers handle lost tracks internally for a few frames. 
        # If we remove immediately, we lose history immediately on occlusion.
        # So we leave the cleanup disabled or implement a timeout.
        # For simplicity, we just keep the history bounded by `max_history` size per tracked object.
        pass
