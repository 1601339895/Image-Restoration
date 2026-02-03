import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization_utils import apply_academic_style, normalize_for_display, compute_psnr, save_figure, plot_image
from extract_attention import extract_from_adaptive_model, extract_from_baseline_model
from utils.model_loader import load_adaptive_model, load_baseline_model
from utils.data_loader import load_test_images

def generate_fig1(adaptive_model_path, baseline_model_path, data_dir, output_dir, device='cuda'):
    """Generate Figure 1: Attention Mechanism Overview"""
    apply_academic_style()

    # Load models
    adaptive_model = load_adaptive_model(adaptive_model_path, device)
    baseline_model = load_baseline_model(baseline_model_path, device)

    # Load one rain image
    rain_images = load_test_images(data_dir, 'rain', num_images=1)
    if not rain_images:
        print("No rain images found!")
        return

    image, img_path = rain_images[0]

    # Extract from both models
    adaptive_result = extract_from_adaptive_model(adaptive_model, image, target_layer_idx=2)
    baseline_result = extract_from_baseline_model(baseline_model, image, target_layer_idx=2)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Row 1: Standard Attention
    # (a) Input
    plot_image(image, axes[0, 0], 'Input Image')

    # (b) Attention Map (averaged across heads)
    if baseline_result['attention_maps']:
        attn = baseline_result['attention_maps'][0][0]  # [heads, HW, HW]
        attn_avg = attn.mean(dim=0).numpy()  # [HW, HW]
        H = W = int(np.sqrt(attn_avg.shape[0]))
        attn_spatial = attn_avg.mean(axis=1).reshape(H, W)

        im = axes[0, 1].imshow(attn_spatial, cmap='viridis')
        axes[0, 1].set_title('Attention Map\n(Standard)', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # (c) Temperature (fixed for standard)
    axes[0, 2].bar([0], [1.0], color='gray')
    axes[0, 2].set_title('Temperature\n(Fixed)', fontsize=10, fontweight='bold')
    axes[0, 2].set_ylabel('Temperature', fontsize=9)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_ylim([0, 2])

    # (d) Output
    plot_image(baseline_result['output'][0], axes[0, 3], 'Restored Output\n(Standard)')

    # Row 2: Adaptive Attention
    # (a) Input
    plot_image(image, axes[1, 0], 'Input Image')

    # (b) Attention Map (averaged across heads)
    if adaptive_result['attention_maps']:
        attn = adaptive_result['attention_maps'][0][0]  # [heads, HW, HW]
        attn_avg = attn.mean(dim=0).numpy()  # [HW, HW]
        H = W = int(np.sqrt(attn_avg.shape[0]))
        attn_spatial = attn_avg.mean(axis=1).reshape(H, W)

        im = axes[1, 1].imshow(attn_spatial, cmap='viridis')
        axes[1, 1].set_title('Attention Map\n(Adaptive)', fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # (c) Temperature (per-head for adaptive)
    if adaptive_result['temperatures']:
        temps = adaptive_result['temperatures'][0][0].squeeze().numpy()  # [heads]
        num_heads = len(temps)
        axes[1, 2].bar(range(num_heads), temps, color='steelblue')
        axes[1, 2].set_title('Temperature\n(Adaptive)', fontsize=10, fontweight='bold')
        axes[1, 2].set_xlabel('Head Index', fontsize=9)
        axes[1, 2].set_ylabel('Temperature', fontsize=9)
        axes[1, 2].set_ylim([0, max(temps) * 1.2])

    # (d) Output
    plot_image(adaptive_result['output'][0], axes[1, 3], 'Restored Output\n(Adaptive)')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig1_attention_overview.pdf')
    save_figure(fig, output_path)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive_model', required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--baseline_model', required=True, help='Path to baseline model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    generate_fig1(args.adaptive_model, args.baseline_model, args.data_dir, args.output_dir, args.device)
