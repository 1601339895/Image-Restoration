"""
Enhanced Evaluation Script for All-in-One Image Restoration
Computes PSNR and SSIM metrics on benchmark datasets
"""

import os
import sys
import argparse
import pathlib
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Import skimage metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("⚠️  Warning: scikit-image not found. Installing...")
    os.system("pip install scikit-image")
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

from net.model_improved import ImprovedLoRaGateRestormer, apply_enhanced_lora_strategy


# ==============================================================================
# Dataset Classes for Evaluation
# ==============================================================================
class TestDataset(Dataset):
    """Generic test dataset for paired clean/degraded images."""
    def __init__(self, degraded_dir, clean_dir=None, transform=None):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform

        # Get file lists
        self.degraded_files = sorted([
            f for f in os.listdir(degraded_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if clean_dir is not None:
            self.clean_files = sorted([
                f for f in os.listdir(clean_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
            assert len(self.degraded_files) == len(self.clean_files), \
                "Mismatch between degraded and clean image counts"

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        # Load degraded image
        deg_path = os.path.join(self.degraded_dir, self.degraded_files[idx])
        deg_img = Image.open(deg_path).convert('RGB')

        if self.transform:
            deg_tensor = self.transform(deg_img)
        else:
            deg_tensor = transforms.ToTensor()(deg_img)

        # Load clean image if available
        if self.clean_dir is not None:
            clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
            clean_img = Image.open(clean_path).convert('RGB')

            if self.transform:
                clean_tensor = self.transform(clean_img)
            else:
                clean_tensor = transforms.ToTensor()(clean_img)

            return deg_tensor, clean_tensor, self.degraded_files[idx]
        else:
            return deg_tensor, None, self.degraded_files[idx]


class BenchmarkDatasets:
    """
    Configuration for standard benchmark datasets.
    Matches the paper's evaluation protocol.
    """
    DATASETS = {
        # Denoising
        'BSD68': {
            'type': 'denoising',
            'sigma': [15, 25, 50]
        },
        # Deraining
        'Rain100L': {
            'type': 'deraining',
            'degraded': 'rainy',
            'clean': 'no_rain'
        },
        # Dehazing
        'SOTS': {
            'type': 'dehazing',
            'degraded': 'hazy',
            'clean': 'clear'
        },
        # Deblurring
        'GoPro': {
            'type': 'deblurring',
            'degraded': 'blur',
            'clean': 'sharp'
        },
        # Low-light enhancement
        'LOL': {
            'type': 'lol',
            'degraded': 'low',
            'clean': 'high'
        }
    }


# ==============================================================================
# Metric Computation
# ==============================================================================
def calculate_psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR between two images."""
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to [0, 255] if needed
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    return psnr(img1, img2, data_range=255.0)


def calculate_ssim(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate SSIM between two images."""
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to [0, 255] if needed
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    return ssim(img1, img2, data_range=255.0, channel_axis=2, multichannel=True)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


# ==============================================================================
# Model Loading
# ==============================================================================
def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"\n📦 Loading model from: {checkpoint_path}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"   Model config: dim={hparams.get('dim', 48)}, "
              f"gate_type={hparams.get('gate_type', 'elementwise')}")

        # Create model
        model = ImprovedLoRaGateRestormer(
            dim=hparams.get('dim', 48),
            num_blocks=hparams.get('num_blocks', [4, 6, 6, 8]),
            num_refinement_blocks=hparams.get('num_refinement_blocks', 4),
            heads=hparams.get('heads', [1, 2, 4, 8]),
            gate_type=hparams.get('gate_type', 'elementwise'),
            use_dfrg=hparams.get('use_dfrg', True)
        )

        # Apply LoRa if used during training
        if 'LoRa_ffn_ratio' in hparams:
            model = apply_enhanced_lora_strategy(
                model,
                lora_ffn_ratio=hparams.get('LoRa_ffn_ratio', 0.5),
                lora_attn_ratio=hparams.get('LoRa_attn_ratio', 0.8)
            )

        # Load state dict
        state_dict = checkpoint['state_dict']
        # Remove 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)
    else:
        raise ValueError("Checkpoint does not contain hyperparameters")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params / 1e6:.2f} M\n")

    return model


# ==============================================================================
# Evaluation Function
# ==============================================================================
@torch.no_grad()
def evaluate_model(model, dataloader, save_dir=None, device='cuda'):
    """
    Evaluate model on a dataset.

    Args:
        model: Restoration model
        dataloader: Test data loader
        save_dir: Directory to save restored images (optional)
        device: Device to run on

    Returns:
        Dictionary with PSNR and SSIM statistics
    """
    model.eval()

    psnr_values = []
    ssim_values = []

    if save_dir:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(dataloader, desc="Evaluating")

    for batch in progress_bar:
        if len(batch) == 3:
            deg_img, clean_img, filename = batch
            has_gt = True
        else:
            deg_img, filename = batch
            clean_img = None
            has_gt = False

        deg_img = deg_img.to(device)
        if has_gt:
            clean_img = clean_img.to(device)

        # Restore image
        restored = model(deg_img)

        # Clamp to valid range
        restored = torch.clamp(restored, 0.0, 1.0)

        # Calculate metrics if ground truth available
        if has_gt:
            for i in range(restored.shape[0]):
                restored_np = tensor_to_numpy(restored[i])
                clean_np = tensor_to_numpy(clean_img[i])

                psnr_val = calculate_psnr(restored_np, clean_np, crop_border=0)
                ssim_val = calculate_ssim(restored_np, clean_np, crop_border=0)

                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)

                # Update progress bar
                progress_bar.set_postfix({
                    'PSNR': f'{np.mean(psnr_values):.2f}',
                    'SSIM': f'{np.mean(ssim_values):.4f}'
                })

        # Save restored images
        if save_dir:
            for i in range(restored.shape[0]):
                restored_np = tensor_to_numpy(restored[i])
                restored_np = (restored_np * 255.0).clip(0, 255).astype(np.uint8)

                fname = filename[i] if isinstance(filename, (list, tuple)) else filename
                save_path = os.path.join(save_dir, fname)

                Image.fromarray(restored_np).save(save_path)

    # Compute statistics
    results = {}
    if psnr_values:
        results['psnr_mean'] = np.mean(psnr_values)
        results['psnr_std'] = np.std(psnr_values)
        results['ssim_mean'] = np.mean(ssim_values)
        results['ssim_std'] = np.std(ssim_values)
        results['num_images'] = len(psnr_values)

    return results


# ==============================================================================
# Main Evaluation Script
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate image restoration model")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--degraded_dir', type=str, required=True,
                       help='Directory with degraded images')
    parser.add_argument('--clean_dir', type=str, default=None,
                       help='Directory with clean images (for metrics)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save restored images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--benchmark', type=str, default=None,
                       choices=list(BenchmarkDatasets.DATASETS.keys()),
                       help='Benchmark dataset name')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("Image Restoration Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Degraded images: {args.degraded_dir}")
    print(f"Clean images: {args.clean_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    # Create dataset
    dataset = TestDataset(
        degraded_dir=args.degraded_dir,
        clean_dir=args.clean_dir,
        transform=transforms.ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )

    print(f"📊 Evaluating on {len(dataset)} images...\n")

    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        save_dir=args.save_dir,
        device=args.device
    )

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    if results:
        print(f"Number of images: {results['num_images']}")
        print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    else:
        print("No ground truth provided, metrics not computed.")
        print(f"Restored images saved to: {args.save_dir}")

    print("=" * 70)

    # Save results to file
    if args.save_dir and results:
        results_file = os.path.join(args.save_dir, 'metrics.txt')
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.degraded_dir}\n")
            f.write(f"Number of images: {results['num_images']}\n")
            f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
            f.write(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n")
        print(f"\n📝 Results saved to: {results_file}")


if __name__ == '__main__':
    main()
