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
        # self.yolo_model = YOLO("../models/yolov8m-v1-300/weights/best.pt")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def identify(self, image: bytes, caption: str) -> List[int]:
        img = Image.open(io.BytesIO(image))
        
        # Detect all objects in image
        yolo_result = self.yolo_model(img, half=True, conf=0.1, verbose=False, classes=[i for i in range(80, 86)])[0]
        yolo_annotations = []
        for box, cls in zip(yolo_result.boxes.xywh, yolo_result.boxes.cls):
            x, y, w, h = box.tolist()
            # Crop bounding boxes from orginal image
            l, t, w, h = int(x-(w/2)), int(y-(h/2)), int(w), int(h)
            cropped = yolo_result.orig_img[t:t+h, l:l+w, ::-1] # BGR to RGB
            class_id = cls.int().item()
            yolo_annotations.append({"image": cropped, "bbox": [l, t, w, h], "class": f"{self.yolo_model.names[class_id]}", "class_id": class_id})

        # cap_class = self.extract_object_class(caption, return_name=True)
        if len(yolo_annotations) == 0:
            return [0, 0, 0, 0]
        
        # Sort annotations by class_id, then by bbox left-top position
        yolo_annotations.sort(key=lambda x: (x["class_id"], x["bbox"][0], x["bbox"][1]))
        
        # Predict the best match for the caption
        inputs = self.clip_processor(text=caption, images=[a["image"] for a in yolo_annotations], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # image-text similarity score

        # Apply bias to logits based on class_id
        cap_class_id = self.extract_object_class(caption)
        bias_factors = torch.ones_like(logits_per_image)
        bias_indexes = [idx for idx, anno in enumerate(yolo_annotations) if anno["class_id"] == cap_class_id]
        if len(bias_indexes) > 0:
            bias_factors[bias_indexes] = 1.2
            logits_biased = logits_per_image * bias_factors
        else:
            logits_biased = logits_per_image

        best_match_idx = logits_biased.argmax(dim=0).item()

        # Debug prints
        # print("{:2} {:13} {:6} ({:>5}) = {:6} [ {:>4},{:>4},{:>4},{:>4} ]".format("ID", "Class", "LogitR", "xBias", "LogitN", "L", "T", "W", "H"))
        # for idx, anno_logits_bias in enumerate(zip(yolo_annotations, logits_per_image.tolist(), bias_factors.tolist())):
        #     anno, logit, bias = anno_logits_bias
        #     print("{:2} {:13} {:.3f} (x{:.2f}) = {:.3f} [ {:4},{:4},{:4},{:4} ]"
        #           .format(anno["class_id"], anno["class"], logit[0], bias[0], logit[0] * bias[0], *anno["bbox"]), end="")
        #     print(f" <== Predicted (Index: {idx})") if idx == best_match_idx else print()
        
        return yolo_annotations[best_match_idx]["bbox"]
    
    def extract_object_class(self, caption, return_name=False):
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
        return self.yolo_model.names[class_id] if return_name else class_id

## Check code
# import base64
# vlm_manager = VLMManager()
# with open("example.jpg", "rb") as file:
#     image = file.read()
# image_bytes = base64.b64encode(image).decode("ascii")
# image_byte = base64.b64decode(image_bytes)

# result = vlm_manager.identify(image_byte, "yellow helicopter")
# print(result)
