from models.clip_model import CLIPModelWrapper
from models.caption_matcher import CaptionMatcher
from utils.image_processor import ImageProcessor
from configs.captions import CANDIDATE_CAPTIONS

def image_captioning(image_path, candidate_captions=CANDIDATE_CAPTIONS):
    clip_model = CLIPModelWrapper("openai/clip-vit-base-patch32")
    
    image_processor = ImageProcessor()
    inputs, _ = image_processor.load_and_preprocess(image_path, "openai/clip-vit-base-patch32")
    image_features = clip_model.get_image_features(inputs)
    
    best_captions, similarities = CaptionMatcher.match_captions(
        image_features, 
        candidate_captions, 
        clip_model
    )
    
    return best_captions, similarities

if __name__ == "__main__":
    image_path = "configs/aman.webp"
    best_captions, similarities = image_captioning(image_path)
    
    top_n = min(5, len(best_captions))
    print(f"Top {top_n} best captions:")
    for i, (caption, similarity) in enumerate(zip(best_captions[:top_n], similarities[:top_n])):
        print(f"{i+1}. {caption} (Similarity: {similarity:.4f})")