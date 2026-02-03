import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization_utils import apply_academic_style, save_figure, plot_image
from extract_attention import extract_from_adaptive_model
from utils.model_loader import load_adaptive_model
from utils.data_loader import load_test_images

def generate_fig3(adaptive_model_path, data_dir, output_dir, device='cuda'):
    """Generate Figure 3: Multi-Degradation Attention Comparison"""
    apply_academic_style()

    # Load model
    model = load_adaptive_model(adaptive_model_path, device)

    # Load one image per degradation type
    degradation_types = ['rain', 'fog', 'noise']
    images_by_deg = {}

    for deg_type in degradation_types:
        images = load_test_images(data_dir, deg_type, num_images=1)
        if images:
            images_by_deg[deg_type] = images[0]
        else:
            print(f"No {deg_type} images found!")

    if len(images_by_deg) == 0:
        print("No images found!")
        return

    # Create figure: 3 rows × 5 columns
    fig, axes = plt.subplots(len(images_by_deg), 5, figsize=(15, 9))

    for row_idx, (deg_type, (image, img_path)) in enumerate(images_by_deg.items()):
        # Extract attention
        result = extract_from_adaptive_model(model, image, target_layer_idx=2)

        # (a) Input
        plot_image(image, axes[row_idx, 0], f'{deg_type.capitalize()}\nInput')

        # Extract attention maps for specific heads
        if result['attention_maps']:
            attn = result['attention_maps'][0][0]  # [heads, HW, HW]
            num_heads = attn.shape[0]

            # Select heads to visualize (1, 4, 8 -> indices 0, 3, 7)
            head_indices = [0, min(3, num_heads-1), min(7, num_heads-1)]
            head_labels = ['Head 1', 'Head 4', 'Head 8']

            for col_idx, (head_idx, head_label) in enumerate(zip(head_indices, head_labels), start=1):
                attn_head = attn[head_idx].numpy()  # [HW, HW]
                H = W = int(np.sqrt(attn_head.shape[0]))
                attn_spatial = attn_head.mean(axis=1).reshape(H, W)

                im = axes[row_idx, col_idx].imshow(attn_spatial, cmap='viridis')
                axes[row_idx, col_idx].set_title(head_label, fontsize=10, fontweight='bold')
                axes[row_idx, col_idx].axis('off')

        # (e) Restored Output
        plot_image(result['output'][0], axes[row_idx, 4], 'Restored')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig3_multideg_comparison.pdf')
    save_figure(fig, output_path)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive_model', required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    generate_fig3(args.adaptive_model, args.data_dir, args.output_dir, args.device)
