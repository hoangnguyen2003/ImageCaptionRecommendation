from PIL import Image
from transformers import CLIPProcessor

class ImageProcessor:
    @staticmethod
    def load_and_preprocess(image_path, clip_model_name):
        image = Image.open(image_path).convert("RGB")
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        inputs = processor(images=image, return_tensors="pt")
        return inputs, processor