import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visualization_utils import apply_academic_style, save_figure, get_degradation_colors
from extract_attention import extract_from_adaptive_model, batch_extract
from utils.model_loader import load_adaptive_model
from utils.data_loader import load_test_images

def generate_fig2(adaptive_model_path, data_dir, output_dir, num_images=50, device='cuda'):
    """Generate Figure 2: Temperature Coefficient Analysis"""
    apply_academic_style()

    # Load model
    model = load_adaptive_model(adaptive_model_path, device)

    # Load images for each degradation type
    degradation_types = ['rain', 'fog', 'noise']
    all_temps_by_deg = {deg: [] for deg in degradation_types}
    all_temps_by_layer = {deg: {0: [], 1: [], 2: [], 3: []} for deg in degradation_types}

    for deg_type in degradation_types:
        images = load_test_images(data_dir, deg_type, num_images=num_images)
        if not images:
            print(f"No {deg_type} images found!")
            continue

        # Extract from multiple layers
        for layer_idx in range(4):
            for image, _ in images:
                result = extract_from_adaptive_model(model, image, target_layer_idx=layer_idx)
                if result['temperatures']:
                    temps = result['temperatures'][0][0].squeeze().numpy()  # [heads]
                    all_temps_by_deg[deg_type].extend(temps)
                    all_temps_by_layer[deg_type][layer_idx].extend(temps)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Temperature distribution across heads (box plot)
    colors = get_degradation_colors()
    data_for_boxplot = []
    labels = []
    positions = []
    pos = 0

    for deg_type in degradation_types:
        if all_temps_by_deg[deg_type]:
            data_for_boxplot.append(all_temps_by_deg[deg_type])
            labels.append(deg_type.capitalize())
            positions.append(pos)
            pos += 1

    bp = axes[0].boxplot(data_for_boxplot, positions=positions, widths=0.6, patch_artist=True)
    for patch, deg_type in zip(bp['boxes'], degradation_types):
        patch.set_facecolor(colors.get(deg_type, 'gray'))

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Temperature', fontsize=10)
    axes[0].set_title('(a) Temperature Distribution', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # (b) Temperature heatmap across layers
    heatmap_data = np.zeros((len(degradation_types), 4))
    for i, deg_type in enumerate(degradation_types):
        for layer_idx in range(4):
            if all_temps_by_layer[deg_type][layer_idx]:
                heatmap_data[i, layer_idx] = np.mean(all_temps_by_layer[deg_type][layer_idx])

    im = axes[1].imshow(heatmap_data, cmap='coolwarm', aspect='auto')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(['Enc1', 'Enc2', 'Enc3', 'Latent'])
    axes[1].set_yticks(range(len(degradation_types)))
    axes[1].set_yticklabels([d.capitalize() for d in degradation_types])
    axes[1].set_title('(b) Temperature Across Layers', fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(degradation_types)):
        for j in range(4):
            text = axes[1].text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=axes[1], fraction=0.046, label='Temperature')

    # (c) Temperature evolution (line plot)
    for deg_type in degradation_types:
        layer_means = []
        layer_stds = []
        for layer_idx in range(4):
            if all_temps_by_layer[deg_type][layer_idx]:
                layer_means.append(np.mean(all_temps_by_layer[deg_type][layer_idx]))
                layer_stds.append(np.std(all_temps_by_layer[deg_type][layer_idx]))
            else:
                layer_means.append(0)
                layer_stds.append(0)

        axes[2].plot(range(4), layer_means, marker='o', label=deg_type.capitalize(),
                    color=colors.get(deg_type, 'gray'), linewidth=2)
        axes[2].fill_between(range(4),
                            np.array(layer_means) - np.array(layer_stds),
                            np.array(layer_means) + np.array(layer_stds),
                            alpha=0.2, color=colors.get(deg_type, 'gray'))

    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(['Enc1', 'Enc2', 'Enc3', 'Latent'])
    axes[2].set_xlabel('Layer', fontsize=10)
    axes[2].set_ylabel('Mean Temperature', fontsize=10)
    axes[2].set_title('(c) Temperature Evolution', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig2_temperature_analysis.pdf')
    save_figure(fig, output_path)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive_model', required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images per degradation type')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    generate_fig2(args.adaptive_model, args.data_dir, args.output_dir, args.num_images, args.device)
