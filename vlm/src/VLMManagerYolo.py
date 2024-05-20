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
        self.yolo_model = YOLO("./models/yolov8/weights/best.pt")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def identify(self, image: bytes, caption: str) -> List[int]:
        img = np.array(Image.open(io.BytesIO(image)))
        
        # Detect all objects in image
        yolo_result = self.yolo_model(img, half=True, conf=0.1)[0]
        yolo_annotations = []

        for box, cls in zip(yolo_result.boxes.xywh, yolo_result.boxes.cls):
            # Box is a Tensor of [x, y, w, h]
            x, y, w, h = box.tolist()
            # Crop bounding boxes from orginal image
            left, top, w, h = int(x-(w/2)), int(y-(h/2)), int(w), int(h)
            cropped = yolo_result.orig_img[top:top+h, left:left+w, ::-1] # BGR to RGB
            obj_class = self.yolo_model.names[cls.item()]
            #print(f"[{left}, {top}, {int(w)}, {int(h)}] {cls.int().item()} {obj_class}")
            yolo_annotations.append({"image": cropped, "bbox": [left, top, w, h], "class": f"{obj_class}"})

        if len(yolo_annotations) == 0:
            return [0, 0, 0, 0]
        
        inputs = self.clip_processor(text=caption, images=[a["image"] for a in yolo_annotations], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # image-text similarity score
        best_match_idx = logits_per_image.argmax(dim=0).item()

        return yolo_annotations[best_match_idx]["bbox"]

## Check code
# import base64
# vlm_manager = VLMManager()
# with open("example.jpg", "rb") as file:
#     image = file.read()
# image_bytes = base64.b64encode(image).decode("ascii")
# image_byte = base64.b64decode(image_bytes)

# result = vlm_manager.identify(image_byte, "yellow helicopter")
# print(result)