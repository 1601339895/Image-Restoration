import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

def apply_academic_style():
    """Apply academic publication style to matplotlib"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 12
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42

def normalize_for_display(tensor):
    """Normalize tensor to [0, 1] for visualization"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = tensor.astype(np.float32)
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val > 1e-5:
        return (tensor - min_val) / (max_val - min_val)
    return tensor

def compute_psnr(img1, img2, max_val=1.0):
    """Compute PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * np.log10(max_val / np.sqrt(mse))

def compute_ssim(img1, img2):
    """Simplified SSIM computation"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)

    sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 5) - mu1 * mu1
    sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 5) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

def compute_attention_entropy(attn_map):
    """Compute entropy of attention map"""
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    attn_map = attn_map + 1e-10
    entropy = -np.sum(attn_map * np.log(attn_map), axis=-1)
    return entropy

def compute_effective_rank(attn_map):
    """Compute effective rank of attention map (exp of entropy)"""
    entropy = compute_attention_entropy(attn_map)
    return np.exp(entropy)

def compute_attention_sparsity(attn_map, threshold=0.01):
    """Compute sparsity of attention map (percentage below threshold)"""
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    return np.mean(attn_map < threshold) * 100

def compute_focus_score(attn_map):
    """Compute focus score (max attention weight)"""
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    return np.max(attn_map, axis=-1)

def save_figure(fig, filename, dpi=300):
    """Save figure in both PDF and PNG formats"""
    fig.savefig(filename.replace('.pdf', '.pdf'), dpi=dpi, bbox_inches='tight', format='pdf')
    fig.savefig(filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight', format='png')
    print(f"Saved: {filename}")

def plot_heatmap(data, ax, title, xlabel, ylabel, cmap='viridis', vmin=None, vmax=None, cbar_label=''):
    """Plot heatmap with colorbar"""
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    return im

def plot_image(img, ax, title):
    """Plot image"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)

    img = normalize_for_display(img)
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def compute_tsne(embeddings, n_components=2, perplexity=30):
    """Compute t-SNE embedding"""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(embeddings)-1),
                random_state=42, n_iter=1000)
    return tsne.fit_transform(embeddings)

def get_degradation_colors():
    """Get consistent colors for degradation types"""
    return {
        'rain': '#3498db',    # Blue
        'fog': '#95a5a6',     # Gray
        'noise': '#e74c3c',   # Red
        'clean': '#2ecc71'    # Green
    }
