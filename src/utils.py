import cv2
import logging
import sys
from typing import Tuple

def setup_logger(name: str = "tracker_app") -> logging.Logger:
    """Sets up a console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def get_video_properties(video_path: str) -> Tuple[cv2.VideoCapture, float, int, int]:
    """
    Opens a video file and retrieves its properties.
    
    Args:
        video_path (str): Path to the input video.
        
    Returns:
        Tuple containing (VideoCapture object, FPS, width, height)
    """
    logger = setup_logger()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, w, h
    
def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Initializes an OpenCV VideoWriter.
    
    Args:
        output_path (str): Path to save the output video.
        fps (float): Frames per second.
        width (int): Frame width.
        height (int): Frame height.
        
    Returns:
        cv2.VideoWriter object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer
