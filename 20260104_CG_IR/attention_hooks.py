import torch
import torch.nn as nn

class AttentionHook:
    """Hook for capturing intermediate activations from Context_Adaptive_Gated_Attention"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.temperatures = []
        self.gate_scores = []
        self.attention_maps = []
        self.features_before_gate = []
        self.features_after_gate = []
        self.context_embeddings = []

    def __call__(self, module, input, output):
        pass

class TemperatureHook:
    """Capture temperature coefficients"""
    def __init__(self):
        self.temperatures = []

    def __call__(self, module, input, output):
        self.temperatures.append(output.detach().cpu())

    def reset(self):
        self.temperatures = []

class GateHook:
    """Capture gate scores"""
    def __init__(self):
        self.gate_scores = []

    def __call__(self, module, input, output):
        self.gate_scores.append(output.detach().cpu())

    def reset(self):
        self.gate_scores = []

class AttentionMapHook:
    """Capture attention maps"""
    def __init__(self):
        self.attention_maps = []

    def __call__(self, module, input, output):
        self.attention_maps.append(output.detach().cpu())

    def reset(self):
        self.attention_maps = []

class PromptHook:
    """Capture prompt embeddings from Degradation_Aware_Module"""
    def __init__(self):
        self.layer_prompts = []
        self.global_feats = []
        self.spatial_gates = []

    def __call__(self, module, input, output):
        layer_prompts, global_feat = output
        self.layer_prompts.append([p.detach().cpu() for p in layer_prompts])
        self.global_feats.append(global_feat.detach().cpu())

    def reset(self):
        self.layer_prompts = []
        self.global_feats = []
        self.spatial_gates = []

def register_hooks(model, target_layer_idx=2):
    """
    Register hooks to capture intermediate activations

    Args:
        model: DACG_IR model
        target_layer_idx: Which encoder level to visualize (0-3)

    Returns:
        dict of hooks
    """
    hooks = {
        'temperature': TemperatureHook(),
        'gate': GateHook(),
        'attention': AttentionMapHook(),
        'prompt': PromptHook(),
    }

    handles = []

    # Hook for prompt generation
    handle = model.context_net.register_forward_hook(hooks['prompt'])
    handles.append(handle)

    # Hook for attention module at target layer
    if target_layer_idx == 0:
        target_blocks = model.encoder_level1
    elif target_layer_idx == 1:
        target_blocks = model.encoder_level2
    elif target_layer_idx == 2:
        target_blocks = model.encoder_level3
    else:
        target_blocks = model.latent

    # Register hooks on the first block of target layer
    attn_module = target_blocks[0].attn

    # Custom hook to capture temperature, gate, and attention
    def capture_attention_internals(module, input, output):
        # We need to modify the forward pass to expose internals
        pass

    hooks['handles'] = handles
    hooks['target_module'] = attn_module

    return hooks

def remove_hooks(hooks):
    """Remove all registered hooks"""
    if 'handles' in hooks:
        for handle in hooks['handles']:
            handle.remove()
