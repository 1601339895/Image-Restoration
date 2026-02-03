import os
import glob
import torch
from PIL import Image
import torchvision.transforms as transforms

def load_test_images(data_dir, degradation_type, num_images=None):
    """Load test images for a specific degradation type"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Map degradation types to directory patterns
    deg_map = {
        'rain': ['rain', 'derain', 'Rain'],
        'fog': ['fog', 'haze', 'dehaze', 'Fog', 'Haze'],
        'noise': ['noise', 'denoise', 'Noise']
    }

    images = []
    patterns = deg_map.get(degradation_type, [degradation_type])

    for pattern in patterns:
        search_paths = [
            os.path.join(data_dir, f'*{pattern}*', 'input', '*.png'),
            os.path.join(data_dir, f'*{pattern}*', 'input', '*.jpg'),
            os.path.join(data_dir, f'*{pattern}*', '*.png'),
            os.path.join(data_dir, f'*{pattern}*', '*.jpg'),
        ]

        for search_path in search_paths:
            found = glob.glob(search_path)
            if found:
                images.extend(found)
                break

    if num_images:
        images = images[:num_images]

    loaded_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        loaded_images.append((img_tensor, img_path))

    return loaded_images

def get_image_pairs(data_dir, degradation_type):
    """Get degraded/clean image pairs"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pairs = []
    deg_map = {
        'rain': ['rain', 'derain'],
        'fog': ['fog', 'haze', 'dehaze'],
        'noise': ['noise', 'denoise']
    }

    patterns = deg_map.get(degradation_type, [degradation_type])

    for pattern in patterns:
        input_dir = os.path.join(data_dir, f'*{pattern}*', 'input')
        target_dir = os.path.join(data_dir, f'*{pattern}*', 'target')

        input_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))

        for input_path in input_paths:
            basename = os.path.basename(input_path)
            target_path = os.path.join(target_dir, basename)

            if os.path.exists(target_path):
                input_img = Image.open(input_path).convert('RGB')
                target_img = Image.open(target_path).convert('RGB')

                input_tensor = transform(input_img)
                target_tensor = transform(target_img)

                pairs.append((input_tensor, target_tensor, input_path))

    return pairs
