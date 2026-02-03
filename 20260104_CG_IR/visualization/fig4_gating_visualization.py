import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization_utils import apply_academic_style, save_figure
from extract_attention import extract_from_adaptive_model
from utils.model_loader import load_adaptive_model
from utils.data_loader import load_test_images

def generate_fig4(adaptive_model_path, data_dir, output_dir, device='cuda'):
    """Generate Figure 4: Gating Mechanism Visualization"""
    apply_academic_style()

    # Load model
    model = load_adaptive_model(adaptive_model_path, device)

    # Load one rain and one noise image
    rain_images = load_test_images(data_dir, 'rain', num_images=1)
    noise_images = load_test_images(data_dir, 'noise', num_images=1)

    images_by_deg = {}
    if rain_images:
        images_by_deg['rain'] = rain_images[0]
    if noise_images:
        images_by_deg['noise'] = noise_images[0]

    if len(images_by_deg) == 0:
        print("No images found!")
        return

    # Create figure: 2 rows × 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row_idx, (deg_type, (image, img_path)) in enumerate(images_by_deg.items()):
        # Extract gating information
        result = extract_from_adaptive_model(model, image, target_layer_idx=2)

        if not result['gate_scores']:
            continue

        # Get gate scores: [B, heads, head_dim, 1]
        gate_scores = result['gate_scores'][0][0].squeeze().numpy()  # [heads, head_dim]

        # (a) Gate scores heatmap
        im = axes[row_idx, 0].imshow(gate_scores.T, cmap='plasma', aspect='auto', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'{deg_type.capitalize()}\nGate Scores', fontsize=10, fontweight='bold')
        axes[row_idx, 0].set_xlabel('Head Index', fontsize=9)
        axes[row_idx, 0].set_ylabel('Channel Index', fontsize=9)
        plt.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)

        # (b) Gate distribution (histogram)
        gate_flat = gate_scores.flatten()
        axes[row_idx, 1].hist(gate_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[row_idx, 1].set_title(f'{deg_type.capitalize()}\nGate Distribution', fontsize=10, fontweight='bold')
        axes[row_idx, 1].set_xlabel('Gate Value', fontsize=9)
        axes[row_idx, 1].set_ylabel('Frequency', fontsize=9)
        axes[row_idx, 1].axvline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold')
        axes[row_idx, 1].legend(fontsize=8)
        axes[row_idx, 1].grid(alpha=0.3)

        # (c) Top-k activated channels
        channel_activation = gate_scores.mean(axis=0)  # Average across heads
        top_k = 10
        top_indices = np.argsort(channel_activation)[-top_k:][::-1]
        top_values = channel_activation[top_indices]

        axes[row_idx, 2].barh(range(top_k), top_values, color='coral')
        axes[row_idx, 2].set_yticks(range(top_k))
        axes[row_idx, 2].set_yticklabels([f'Ch {i}' for i in top_indices])
        axes[row_idx, 2].set_title(f'{deg_type.capitalize()}\nTop-{top_k} Channels', fontsize=10, fontweight='bold')
        axes[row_idx, 2].set_xlabel('Mean Gate Score', fontsize=9)
        axes[row_idx, 2].invert_yaxis()
        axes[row_idx, 2].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig4_gating_visualization.pdf')
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

    generate_fig4(args.adaptive_model, args.data_dir, args.output_dir, args.device)
