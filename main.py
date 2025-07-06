from models.clip_model import CLIPModelWrapper
from models.caption_matcher import CaptionMatcher
from utils.image_utils import load_image
from configs.captions import CANDIDATE_CAPTIONS
import argparse

def image_captioning(image_name, candidate_captions=CANDIDATE_CAPTIONS):
    clip_model = CLIPModelWrapper('openai/clip-vit-base-patch32')
    image_features = clip_model.get_image_features(load_image(image_name))
    
    best_captions, similarities = CaptionMatcher.match_captions(
        image_features,
        candidate_captions,
        clip_model
    )
    
    return best_captions, similarities

def main(image_name):
    best_captions, similarities = image_captioning(image_name)
    
    top_n = min(5, len(best_captions))
    print(f'Top {top_n} best captions:')
    for i, (caption, similarity) in enumerate(zip(best_captions[:top_n], similarities[:top_n])):
        print(f'{i+1} - {caption} (similarity: {similarity:.5f})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Caption Recommendation System'
    )
    parser.add_argument(
        '--image_name',
        type=str,
        help='Image file name',
        default='dog.jpg'
    )

    main(parser.parse_args().image_name)