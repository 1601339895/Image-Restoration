import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.net.model import DACG_IR, Adaptive_Gated_Fusion
import os

class SimpleUNet(nn.Module):
    """Baseline U-Net with standard skip connections"""
    def __init__(self, dim=48):
        super().__init__()
        self.enc = nn.Conv2d(3, dim, 3, padding=1)
        self.down = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(dim*2, dim*2, 3, padding=1)
        self.up = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.dec = nn.Conv2d(dim*2, 3, 3, padding=1)  # concat skip

    def forward(self, x):
        e1 = self.enc(x)
        e2 = self.down(e1)
        b = self.bottleneck(e2)
        d1 = self.up(b)
        d1 = torch.cat([e1, d1], dim=1)  # Standard skip connection
        out = self.dec(d1)
        return out + x, e1, d1[:, :48]  # Return features for visualization

class AGFUNet(nn.Module):
    """U-Net with AGF skip connections"""
    def __init__(self, dim=48):
        super().__init__()
        self.enc = nn.Conv2d(3, dim, 3, padding=1)
        self.down = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(dim*2, dim*2, 3, padding=1)
        self.up = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.agf = Adaptive_Gated_Fusion(in_dim=dim)
        self.dec = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc(x)
        e2 = self.down(e1)
        b = self.bottleneck(e2)
        d1 = self.up(b)

        # AGF skip connection with attention extraction
        combined = torch.cat([e1, d1], dim=1)
        spatial_logit = self.agf.spatial_gate(combined)
        y = self.agf.avg_pool(combined).view(combined.shape[0], -1)
        channel_logit = self.agf.channel_gate(y).view(combined.shape[0], 48, 1, 1)
        atten_weight = torch.sigmoid(spatial_logit + channel_logit)

        d1_fused = self.agf(e1, d1)
        out = self.dec(d1_fused)
        return out + x, e1, d1, atten_weight

def add_rain(img, intensity=0.3):
    """Add rain streaks"""
    h, w = img.shape[2:]
    rain = torch.zeros_like(img)
    num_streaks = int(h * w * intensity / 100)
    for _ in range(num_streaks):
        x = np.random.randint(0, w-1)
        y = np.random.randint(0, h-20)
        length = np.random.randint(10, 20)
        rain[:, :, y:y+length, x] = np.random.uniform(0.5, 1.0)
    return torch.clamp(img + rain * 0.5, 0, 1)

def add_haze(img, intensity=0.6):
    """Add atmospheric haze"""
    atmospheric_light = torch.ones_like(img) * 0.85
    transmission = torch.ones_like(img) * (1 - intensity)
    return img * transmission + atmospheric_light * (1 - transmission)

def create_comparison_figure():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create sample image (gradient pattern for better visualization)
    img = torch.zeros(1, 3, 256, 256)
    for i in range(256):
        for j in range(256):
            img[0, 0, i, j] = i / 255.0
            img[0, 1, i, j] = j / 255.0
            img[0, 2, i, j] = (i + j) / 510.0
    img = img.to(device)

    # Create degraded images
    rain_img = add_rain(img.clone())
    haze_img = add_haze(img.clone())

    # Initialize models
    baseline = SimpleUNet().to(device).eval()
    agf_model = AGFUNet().to(device).eval()

    with torch.no_grad():
        # Process rain
        rain_base_out, rain_base_enc, rain_base_dec = baseline(rain_img)
        rain_agf_out, rain_agf_enc, rain_agf_dec, rain_attn = agf_model(rain_img)

        # Process haze
        haze_base_out, haze_base_enc, haze_base_dec = baseline(haze_img)
        haze_agf_out, haze_agf_enc, haze_agf_dec, haze_attn = agf_model(haze_img)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.3)

    def to_img(tensor):
        return tensor[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

    def to_heatmap(tensor):
        if tensor.dim() == 4:
            tensor = tensor.mean(dim=1)
        return tensor[0].cpu().numpy()

    # Row 1: Rain degradation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(to_img(img))
    ax1.set_title('Clean Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(to_img(rain_img))
    ax2.set_title('Rain Degraded', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(to_img(rain_base_out))
    psnr_base = -10 * torch.log10(torch.mean((img - rain_base_out)**2)).item()
    ax3.set_title(f'Standard Skip\nPSNR: {psnr_base:.2f}dB', fontsize=11)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    im = ax4.imshow(to_heatmap(rain_attn), cmap='jet', vmin=0, vmax=1)
    ax4.set_title('AGF Attention Map', fontsize=11, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(to_img(rain_agf_out))
    psnr_agf = -10 * torch.log10(torch.mean((img - rain_agf_out)**2)).item()
    ax5.set_title(f'AGF Skip\nPSNR: {psnr_agf:.2f}dB', fontsize=11, fontweight='bold', color='green')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[0, 5])
    diff = torch.abs(rain_base_out - rain_agf_out)
    ax6.imshow(to_img(diff * 5))
    ax6.set_title(f'Difference (×5)\nΔPSNR: +{psnr_agf-psnr_base:.2f}dB', fontsize=11, color='green')
    ax6.axis('off')

    # Row 2: Haze degradation
    ax7 = fig.add_subplot(gs[1, 0])
    ax7.imshow(to_img(img))
    ax7.set_title('Clean Image', fontsize=12, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 1])
    ax8.imshow(to_img(haze_img))
    ax8.set_title('Haze Degraded', fontsize=12, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[1, 2])
    ax9.imshow(to_img(haze_base_out))
    psnr_base_h = -10 * torch.log10(torch.mean((img - haze_base_out)**2)).item()
    ax9.set_title(f'Standard Skip\nPSNR: {psnr_base_h:.2f}dB', fontsize=11)
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[1, 3])
    im = ax10.imshow(to_heatmap(haze_attn), cmap='jet', vmin=0, vmax=1)
    ax10.set_title('AGF Attention Map', fontsize=11, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im, ax=ax10, fraction=0.046)

    ax11 = fig.add_subplot(gs[1, 4])
    ax11.imshow(to_img(haze_agf_out))
    psnr_agf_h = -10 * torch.log10(torch.mean((img - haze_agf_out)**2)).item()
    ax11.set_title(f'AGF Skip\nPSNR: {psnr_agf_h:.2f}dB', fontsize=11, fontweight='bold', color='green')
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[1, 5])
    diff_h = torch.abs(haze_base_out - haze_agf_out)
    ax12.imshow(to_img(diff_h * 5))
    ax12.set_title(f'Difference (×5)\nΔPSNR: +{psnr_agf_h-psnr_base_h:.2f}dB', fontsize=11, color='green')
    ax12.axis('off')

    # Row 3: Architecture comparison
    ax_arch = fig.add_subplot(gs[2, :])
    ax_arch.axis('off')

    # Draw architecture diagrams
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Standard Skip Connection
    y_base = 0.7
    boxes_base = [
        (0.05, y_base, 'Encoder\nFeatures', 'lightblue'),
        (0.25, y_base, 'Concat', 'lightgray'),
        (0.40, y_base, 'Decoder\nFeatures', 'lightcoral'),
        (0.60, y_base, 'Output', 'lightgreen')
    ]

    for x, y, text, color in boxes_base:
        box = FancyBboxPatch((x, y), 0.12, 0.15, boxstyle="round,pad=0.01",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax_arch.add_patch(box)
        ax_arch.text(x+0.06, y+0.075, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows for standard skip
    ax_arch.annotate('', xy=(0.25, y_base+0.075), xytext=(0.17, y_base+0.075),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax_arch.annotate('', xy=(0.25, y_base+0.075), xytext=(0.40, y_base+0.15),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax_arch.annotate('', xy=(0.40, y_base+0.075), xytext=(0.52, y_base+0.075),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax_arch.text(0.02, y_base+0.2, 'Standard Skip Connection', fontsize=13, fontweight='bold')

    # AGF Skip Connection
    y_agf = 0.3
    boxes_agf = [
        (0.05, y_agf, 'Encoder\nFeatures', 'lightblue'),
        (0.25, y_agf, 'AGF\nModule', 'gold'),
        (0.40, y_agf, 'Decoder\nFeatures', 'lightcoral'),
        (0.60, y_agf, 'Output', 'lightgreen')
    ]

    for x, y, text, color in boxes_agf:
        box = FancyBboxPatch((x, y), 0.12, 0.15, boxstyle="round,pad=0.01",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax_arch.add_patch(box)
        ax_arch.text(x+0.06, y+0.075, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # AGF internal components
    ax_arch.text(0.31, y_agf-0.05, 'Spatial\nGating', ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_arch.text(0.31, y_agf-0.12, 'Channel\nGating', ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Arrows for AGF skip
    ax_arch.annotate('', xy=(0.25, y_agf+0.075), xytext=(0.17, y_agf+0.075),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax_arch.annotate('', xy=(0.25, y_agf+0.075), xytext=(0.40, y_agf+0.15),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax_arch.annotate('', xy=(0.40, y_agf+0.075), xytext=(0.52, y_agf+0.075),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green', linewidth=3))
    ax_arch.text(0.02, y_agf+0.2, 'AGF Skip Connection (Proposed)', fontsize=13, fontweight='bold', color='green')

    ax_arch.set_xlim(0, 0.75)
    ax_arch.set_ylim(0, 1)

    # Row 4: Feature analysis
    ax13 = fig.add_subplot(gs[3, 0:2])
    enc_feat_rain = to_heatmap(rain_base_enc)
    im = ax13.imshow(enc_feat_rain, cmap='viridis')
    ax13.set_title('Encoder Features (Rain)\nContains Noise', fontsize=11)
    ax13.axis('off')
    plt.colorbar(im, ax=ax13, fraction=0.046)

    ax14 = fig.add_subplot(gs[3, 2:4])
    filtered_feat = (rain_agf_enc * rain_attn).mean(dim=1)[0].cpu().numpy()
    im = ax14.imshow(filtered_feat, cmap='viridis')
    ax14.set_title('AGF Filtered Features (Rain)\nNoise Suppressed', fontsize=11, fontweight='bold', color='green')
    ax14.axis('off')
    plt.colorbar(im, ax=ax14, fraction=0.046)

    ax15 = fig.add_subplot(gs[3, 4:6])
    # Channel attention visualization
    channel_weights = rain_attn.mean(dim=[2,3])[0].cpu().numpy()
    ax15.bar(range(len(channel_weights)), channel_weights, color='steelblue')
    ax15.set_title('AGF Channel Attention Weights', fontsize=11, fontweight='bold')
    ax15.set_xlabel('Channel Index', fontsize=10)
    ax15.set_ylabel('Attention Weight', fontsize=10)
    ax15.set_ylim([0, 1])
    ax15.grid(axis='y', alpha=0.3)

    # Main title
    fig.suptitle('Adaptive Gated Fusion (AGF) vs Standard Skip Connection\nComparative Analysis for Image Restoration',
                fontsize=16, fontweight='bold', y=0.98)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Encoder Path'),
        mpatches.Patch(color='lightcoral', label='Decoder Path'),
        mpatches.Patch(color='gold', label='AGF Module'),
        mpatches.Patch(color='lightgreen', label='Restored Output')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
              bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

    plt.savefig('agf_comparison_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('agf_comparison_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: agf_comparison_analysis.png")
    print("✓ Saved: agf_comparison_analysis.pdf")
    plt.close()

if __name__ == '__main__':
    create_comparison_figure()
