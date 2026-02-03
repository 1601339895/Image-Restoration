import numpy as np
import torch

def compute_effective_rank(attn):
    """Compute effective rank: exp(entropy)"""
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    attn = attn + 1e-10
    entropy = -np.sum(attn * np.log(attn), axis=-1)
    return np.exp(entropy)

def compute_sparsity(attn, threshold=0.01):
    """Compute sparsity: percentage below threshold"""
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    return np.mean(attn < threshold) * 100

def compute_focus_score(attn):
    """Compute focus score: max attention weight"""
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    return np.max(attn, axis=-1)

def aggregate_statistics(values):
    """Compute mean, std, min, max"""
    values = np.array(values)
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }
