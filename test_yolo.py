import json
import os
import sys
from dotenv import load_dotenv
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from scoring.vlm_eval import bb_iou
from statistics import mean

def parse_truths(truth_path):
    instances = []
    with open(truth_path, "r") as f:
        for line in tqdm(f, desc="Reading vlm.jsonl"):
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            for annotation in instance["annotations"]:
                instances.append(
                    {
                        "image": instance["image"],
                        "caption": annotation["caption"],
                        "bbox": annotation["bbox"],
                    }
                )

    image_file_to_annotations = {}
    for image_file, annotations in groupby(instances, key=lambda x: x["image"]):
        image_file_to_annotations[image_file] = list(annotations)

    return image_file_to_annotations

def extract_object_class(caption):
    # Extract the object's class from the caption
    if "aircraft" in caption: # light, commercial, cargo
        class_id = 80
    elif "drone" in caption:
        class_id = 81
    elif "helicopter" in caption:
        class_id = 82
    elif "fighter plane" in caption:
        class_id = 83
    elif "fighter jet" in caption:
        class_id = 84
    elif "missile" in caption:
        class_id = 85
    else:
        print(f"Unknown class for caption: {caption}")
        return None
    return class_id

def test_yolo(weights_path: str, images_dir: Path, truths: dict[str, list]):
    yolo_model = YOLO(weights_path)

    results = [] # tuple of (image_path, % correct)

    for image_file in tqdm(truths.keys(), desc="Testing YOLO"):
        img = Image.open(images_dir / image_file)
        yolo_result = yolo_model(img, half=True, conf=0.1, verbose=False, classes=[i for i in range(80, 86)])[0]

        truth_annos = truths[image_file][:] # Copy the list of annotations
        correct_count = 0

        for box, cls in zip(yolo_result.boxes.xywh, yolo_result.boxes.cls):
            x, y, w, h = box.tolist()
            l, t, w, h = int(x-(w/2)), int(y-(h/2)), int(w), int(h)
            class_id = cls.int().item()

            # Get all truth annotations that have an IoU > 0.5 with the current box (should be at most 1)
            matching_idxs = [idx for idx, truth in enumerate(truth_annos) 
                             if class_id == extract_object_class(truth["caption"]) and bb_iou(truth["bbox"], [l, t, w, h]) > 0.5]
            
            if len(matching_idxs) > 0:
                # Remove the matched truth annotations
                if len(matching_idxs) > 1:
                    print("Warning: multiple matching truth annotations")
                for idx in matching_idxs:
                    truth_annos.pop(idx)
                correct_count += 1
        
        results.append((image_file, correct_count / len(truths[image_file])))

    return results
                

if __name__ == "__main__":
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    else:
        print("Usage: python test_yolo.py <weights_path>")
        sys.exit(1)

    load_dotenv()
    TEAM_NAME, TEAM_TRACK = os.getenv("TEAM_NAME"), os.getenv("TEAM_TRACK")
    on_gcp = None not in [TEAM_NAME, TEAM_TRACK]
    
    track_dataset_dir = Path(f"{'/home/jupyter/' if on_gcp else ''}advanced")
    truth_jsonl_file = track_dataset_dir / "vlm.jsonl"
    
    truths = parse_truths(truth_jsonl_file)
    results = test_yolo(weights_path, track_dataset_dir / "images", truths)

    accuracy = mean([score for _, score in results])
    print(f"Accuracy:\n{weights_path} {accuracy}")