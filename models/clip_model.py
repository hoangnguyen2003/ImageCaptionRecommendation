import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPModelWrapper:
    def __init__(self, model_name):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
    
    def get_image_features(self, image):
        image_inputs = self.processor(images=image, return_tensors='pt')
        with torch.no_grad():
            return self.model.get_image_features(**image_inputs)
    
    def get_text_features(self, texts):
        text_inputs = self.processor(text=texts, return_tensors='pt', padding=True)
        with torch.no_grad():
            return self.model.get_text_features(**text_inputs)