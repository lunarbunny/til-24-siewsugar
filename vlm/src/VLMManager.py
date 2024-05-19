import numpy as np
import io
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    CLIPProcessor,
    CLIPModel,
)
from typing import List

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VLM] Device: {self.device}")
        self.detr_model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", device_map=self.device)
        self.detr_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", device_map=self.device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map=self.device)

    def detect_objects(self, image):
        with torch.no_grad():
            inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.detr_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.detr_processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]
        return results["boxes"]

    def object_images(self, image, boxes):
        image_arr = np.array(image)
        all_images = []
        for box in boxes:
            x1, y1, x2, y2 = [int(val) for val in box]
            _image = image_arr[y1:y2, x1:x2]
            all_images.append(_image)
        return all_images

    def identify_target(self, labels, images):
        inputs = self.clip_processor(
            text=labels, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        most_similar_idx = logits_per_image.argmax(dim=0).item()
        print(most_similar_idx)
        return most_similar_idx

    def identify(self, image: bytes, caption: str) -> List[int]:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        detected_objects = self.detect_objects(pil_image)
        if len(detected_objects) == 0:
            return [0,0,0,0]
        images = self.object_images(pil_image, detected_objects)
        idx = self.identify_target(caption, images)
        x1, y1, x2, y2 = [int(val) for val in detected_objects[idx].tolist()]
        width = x2 - x1
        height = y2 - y1
        return [x1, y1, width, height]