from typing import List
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch
import io

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VLM] Device: {self.device}")
        self.yolo_model = YOLO("yolov8x.pt")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def identify(self, image: bytes, caption: str) -> List[int]:
        img = np.array(Image.open(io.BytesIO(image)))
        
        # Detect all objects in image
        yolo_result = self.yolo_model(img, conf=0.1)[0]
        yolo_objects = []
        for bbox in yolo_result.boxes.xywh:
            # Box is a Tensor of [x, y, w, h]
            # x, y is the center
            x, y, w, h = bbox.tolist()
            # Crop bounding boxes from orginal image
            top, left = round(y-(h/2)), round(x-(w/2))
            cropped = img[top:int(top+h), left:int(left+w), ::-1]
            yolo_objects.append((cropped, [left, top, int(w), int(h)]))

        if len(yolo_objects) == 0:
            return [0, 0, 0, 0]
        
        inputs = self.clip_processor(text=caption, images=[o[0] for o in yolo_objects], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # image-text similarity score
        best_match_idx = logits_per_image.argmax(dim=0).item()

        return yolo_objects[best_match_idx][1]
