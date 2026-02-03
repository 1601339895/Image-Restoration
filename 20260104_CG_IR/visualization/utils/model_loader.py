import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_adaptive_model(checkpoint_path, device='cuda'):
    """Load DACG_IR model with adaptive attention"""
    from src.net.model import DACG_IR

    model = DACG_IR()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

def load_baseline_model(checkpoint_path, device='cuda'):
    """Load baseline model with standard attention"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        from src.net.model import DACG_IR
        model = DACG_IR()
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model
