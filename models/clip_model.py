import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPModelWrapper:
    def __init__(self, model_name):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_image_features(self, image_inputs):
        with torch.no_grad():
            return self.model.get_image_features(**image_inputs)
    
    def get_text_features(self, text_inputs):
        with torch.no_grad():
            return self.model.get_text_features(**text_inputs)