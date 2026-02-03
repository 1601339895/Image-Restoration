import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def extract_from_adaptive_model(model, image, target_layer_idx=2):
    """Extract attention internals from adaptive model

    Args:
        model: DACG_IR model with adaptive attention
        image: Input tensor [B, 3, H, W] or [3, H, W]
        target_layer_idx: Which encoder level to extract from (0-3)

    Returns:
        dict with keys: output, attention_maps, temperatures, gate_scores, layer_prompts
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    # Storage for extracted values
    extracted = {
        'attention_maps': [],
        'temperatures': [],
        'gate_scores': [],
        'layer_prompts': None,
        'output': None
    }

    # Hook to capture attention internals
    def attention_hook(module, input, output):
        if isinstance(output, tuple) and len(output) == 4:
            out, attn, temp, gate = output
            extracted['attention_maps'].append(attn.detach().cpu())
            extracted['temperatures'].append(temp.detach().cpu())
            extracted['gate_scores'].append(gate.detach().cpu())
            return out
        return output

    # Register hook on target layer
    if target_layer_idx == 0:
        target_blocks = model.encoder_level1
    elif target_layer_idx == 1:
        target_blocks = model.encoder_level2
    elif target_layer_idx == 2:
        target_blocks = model.encoder_level3
    else:
        target_blocks = model.latent

    # Temporarily modify forward to return internals
    original_forward = target_blocks[0].attn.forward

    def modified_forward(x, context_emb):
        return original_forward(x, context_emb, return_internals=True)

    target_blocks[0].attn.forward = modified_forward
    handle = target_blocks[0].attn.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        output = model(image)

    # Restore original forward
    target_blocks[0].attn.forward = original_forward
    handle.remove()

    extracted['output'] = output.detach().cpu()

    return extracted

def extract_from_baseline_model(model, image, target_layer_idx=2):
    """Extract attention maps from baseline model with standard attention

    Args:
        model: Baseline model with standard attention
        image: Input tensor [B, 3, H, W] or [3, H, W]
        target_layer_idx: Which encoder level to extract from (0-3)

    Returns:
        dict with keys: output, attention_maps
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    extracted = {
        'attention_maps': [],
        'output': None
    }

    # Hook to capture attention maps
    def attention_hook(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            out, attn = output
            extracted['attention_maps'].append(attn.detach().cpu())
            return out
        return output

    # Register hook on target layer
    if target_layer_idx == 0:
        target_blocks = model.encoder_level1
    elif target_layer_idx == 1:
        target_blocks = model.encoder_level2
    elif target_layer_idx == 2:
        target_blocks = model.encoder_level3
    else:
        target_blocks = model.latent

    handle = target_blocks[0].attn.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        output = model(image)

    handle.remove()

    extracted['output'] = output.detach().cpu()

    return extracted

def batch_extract(model, image_list, degradation_types, is_adaptive=True, target_layer_idx=2):
    """Process multiple images and collect statistics

    Args:
        model: Model to extract from
        image_list: List of (image_tensor, image_path) tuples
        degradation_types: List of degradation type labels (same length as image_list)
        is_adaptive: Whether model is adaptive or baseline
        target_layer_idx: Which encoder level to extract from

    Returns:
        dict with aggregated statistics
    """
    all_results = []

    for (image, path), deg_type in zip(image_list, degradation_types):
        if is_adaptive:
            result = extract_from_adaptive_model(model, image, target_layer_idx)
        else:
            result = extract_from_baseline_model(model, image, target_layer_idx)

        result['degradation_type'] = deg_type
        result['image_path'] = path
        all_results.append(result)

    return all_results
