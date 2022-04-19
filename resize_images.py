import os
import math
import shutil

from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

# Resize an image.
def preprocess_resize(target_width):
    def preprocess(image: Image.Image, log) -> Image.Image:
        (width, height) = image.size
        ratio = width / height

        if width > target_width:
            target_height = math.floor(target_width / ratio)
            log(f'Resizing: To size {target_width}x{target_height}')
            image = image.resize((target_width, target_height))
        else:
            log('Resizing: Image already resized, skipping...')

        return image
    return preprocess