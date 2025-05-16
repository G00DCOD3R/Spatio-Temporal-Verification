import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple

def get_bounding_box_info_udacity(bounding_box_path: str, json_file_name: str) -> Dict:
    """
    Load bounding box information from a JSON file.

    Args:
        bounding_box_path: Path to the JSON file containing bounding box information
        json_file_name: Name of the JSON file to load

    Returns:
        Dictionary containing bounding box information
    """
    with open(os.path.join(bounding_box_path, json_file_name), "r") as f:
        bounding_box_info = json.load(f)
    return bounding_box_info


def process_bounding_boxes_udacity(
     bounding_box_info: Dict, H: int, W: int, filename: str
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Process bounding box information into frame_patch_locations format.

    Args:
        bounding_box_info: Dictionary containing bounding box information for each object
        H: Height of the video frames
        W: Width of the video frames
        filename: Name of the image file

    Returns:
        Dictionary with frame indices as keys and lists of (x1, y1, x2, y2) tuples as values
    """
    frame_patch_locations = {}

    # Get the frame number from image name
    frame_num = filename
    
    # Process each object's bounding boxes
    for (x1, y1, x2, y2) in bounding_box_info["xyxyn"]:
        # Convert normalized coordinates to pixel coordinates
        x1_px = int(x1 * W)
        y1_px = int(y1 * H)
        x2_px = int(x2 * W)
        y2_px = int(y2 * H)

        # Store the coordinates in frame_patch_locations
        if frame_num not in frame_patch_locations:
            frame_patch_locations[frame_num] = []
        frame_patch_locations[frame_num].append((x1_px, y1_px, x2_px, y2_px))

    return frame_patch_locations 