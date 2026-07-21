from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from PIL import Image
import os
import torch
from pathlib import Path
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = "spatio-temporal-verification/experiments/ucf-101/models/yolo/yolo12x.pt"
VIDEO_FOLDERS_PATH = "/vol/bitbucket/svadakku/data/ucf101_jpegs_256"
OUTPUT_FOLDERS_PATH = "/vol/bitbucket/svadakku/data/ucf101_jpegs_256_tracking/"


class YOLO_Video_Tracking():
    "Characterizes a dataset for PyTorch"
    def __init__(self, model, video_path, output_path):
        "Initialization"
        self.model = model
        self.video_path = video_path
        self.output_path = output_path

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def track(self):
        model = self.model
        # Store the track history
        track_history = defaultdict(lambda: [])
        
        # Obtain the frame count from the video directory
        image_files = [f for f in os.listdir(self.video_path) if f.endswith('.jpg')]
        image_files.sort()
        
        # Store the objects in a dictionary, for each object, store the image file and bounding box
        objects = defaultdict(lambda: [])
        
        
        # Read the images in the video folder
        for image_file in image_files:
            frame = cv2.imread(os.path.join(self.video_path, image_file))
            try:
                # Run YOLO11 tracking on the frame, persisting tracks between frames, increasing the confidence threshold to >= 0.5 
                result = model.track(frame, conf=0.6, persist=True, show=False)[0]

                frame_boxes = defaultdict(lambda: [])
                
                
                
                # Get the boxes and track IDs
                if result.boxes and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    frame_boxes['id'] = result.boxes.id.int().tolist()
                    frame_boxes['xywh'] = result.boxes.xywh.cpu().tolist()
                    frame_boxes['xywhn'] = result.boxes.xywhn.cpu().tolist()
                    frame_boxes['xyxy'] = result.boxes.xyxy.cpu().tolist()
                    frame_boxes['xyxyn'] = result.boxes.xyxyn.cpu().tolist()
                    
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Visualize the result on the frame
                    frame = result.plot()

                    for i in range(len(result.boxes.id)):
                        box = boxes[i]
                        track_id = track_ids[i]
                        
                        # Bounding box information to the object
                        bnd_box = defaultdict(lambda: [])
                        bnd_box['image'] = image_file
                        bnd_box['xywh'] = result.boxes.xywh[i].cpu().tolist()
                        bnd_box['xywhn'] = result.boxes.xywhn[i].cpu().tolist()
                        bnd_box['xyxy'] = result.boxes.xyxy[i].cpu().tolist()
                        bnd_box['xyxyn'] = result.boxes.xyxyn[i].cpu().tolist()
                        bnd_box['conf'] = result.boxes.conf[i].cpu().item()
                        
                        # For each object, store the image file and bounding box
                        if track_id not in objects:
                            obj = defaultdict(lambda: [])
                            obj['id'] = track_id
                            obj['cls'] = result.boxes.cls[i].int().cpu().tolist()
                            obj['name'] = result.names[result.boxes.cls[i].item()]
                            obj['bounding_box'] = []
                            # bounding boxes
                            objects[track_id] = obj
                        
                        obj['bounding_box'].append(bnd_box)
                        # Store the object in the objects dictionary
                        objects[track_id] = obj
                        
                    
                    
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        # if len(track) > 30:  # retain 30 tracks for 30 frames
                        #     track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Display the annotated frame
                # cv2.imshow("YOLO11 Tracking", frame)
                # Split the video file name before jpg to save the output
                image_file_name = image_file.split('.')[0] + '_tracked.jpg'
                image_boxes_file_name = image_file.split('.')[0] + '_tracked.json'
                
                
                cv2.imwrite(os.path.join(self.output_path, image_file_name), frame)
                json.dump(frame_boxes, open(os.path.join(self.output_path, image_boxes_file_name), 'w'), indent=4)

                
            except Exception as e:
                print(f"Error on frame {image_file}: {str(e)}")
                break
        
        # Save the tracked objects in a JSON file
        if(len(objects) > 0):
            objects_tracked_file_name = 'all_objects_bounding_box_tracked.json'
            json.dump(objects, open(os.path.join(self.output_path, objects_tracked_file_name), 'w'), indent=4)
            print(f"Tracked objects saved to {os.path.join(self.output_path, objects_tracked_file_name)}")
            
        # cv2.destroyAllWindows()
        

def folder_names_generator(folders_path):
    "Iterate through all folders in a directory"
    for folder in os.listdir(folders_path):
        folder_path = os.path.join(folders_path, folder)
        if os.path.isdir(folder_path):
            yield folder
            # yield folder_path
        else:
            print(f"Skipping {folder}, not a directory.")
            continue
        
def yolo_tracking_video_folders(folder_names):
    "Iterate through all folders in a directory and track objects in each folder"
    for folder in folder_names_generator(folder_names):
        print(f"Processing folder: {folder}")
        # Load the YOLO12 model
        model = YOLO(MODEL_PATH)

        # Open the video file
        video_path = f'{VIDEO_FOLDERS_PATH}/{folder}'
        output_path = f'{OUTPUT_FOLDERS_PATH}/{folder}_tracking'

        # Create the output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        video_tracking = YOLO_Video_Tracking(model, video_path, output_path)
        video_tracking.track()
        print(f"Finished processing folder: {folder}")

if __name__ == "__main__":
    yolo_tracking_video_folders(VIDEO_FOLDERS_PATH)
    # YOLO_Video_Tracking(video_path, output_path).track()

# YOLO_Video_Tracking(video_path, output_path).track()