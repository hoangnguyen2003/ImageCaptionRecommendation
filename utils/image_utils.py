from PIL import Image
from pathlib import Path

def load_image(image_name):
    image_path = Path(__file__).parent.parent / 'configs' / image_name
    return Image.open(image_path).convert('RGB')