import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


class FreezeStrategy:
    """高级冻结策略管理器，支持多种冻结策略"""
    
    def __init__(self, model: torch.nn.Module, config: Dict, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.strategy = config.get('strategy', 'freeze_all_except_last_layers')
        self.frozen_layers = config.get('frozen_layers', [])
        self.freeze_ratio = config.get('freeze_ratio', 0.8)
        self.layer_mapping = self._create_layer_mapping()
        self.initial_state = {}
        self._save_initial_state()
    
    def _create_layer_mapping(self) -> Dict[str, torch.nn.Module]:
        """创建层名称到模块的映射，支持InfNet架构"""
        layer_mapping = {}
        
        # Stem部分
        layer_mapping['stem'] = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.stem_dw
        )
        
        # ResNet层
        layer_mapping['layer1'] = self.model.layer1
        layer_mapping['layer2'] = self.model.layer2
        layer_mapping['layer3'] = self.model.layer3
        layer_mapping['layer4'] = self.model.layer4
        
        # Head部分
        if hasattr(self.model, 'head_conv') and hasattr(self.model, 'fc'):
            layer_mapping['head'] = torch.nn.Sequential(
                self.model.head_conv,
                self.model.head_bn,
                self.model.fc
            )
        elif hasattr(self.model, 'global_pool') and hasattr(self.model, 'fc'):
            layer_mapping['head'] = torch.nn.Sequential(
                self.model.global_pool,
                self.model.dropout,
                self.model.fc
            )
        
        return layer_mapping
    
    def _save_initial_state(self):
        """保存初始参数状态用于验证"""
        for name, param in self.model.named_parameters():
            self.initial_state[name] = param.data.clone().cpu()
    
    def apply_freeze(self):
        """应用冻结策略"""
        self.logger(f"🔒 Applying freeze strategy: {self.strategy}")
        
        if self.strategy == "layer_wise":
            self._freeze_layer_wise()
        elif self.strategy == "ratio":
            self._freeze_by_ratio()
        elif self.strategy == "freeze_all_except_head":
            self._freeze_all_except_head()
        elif self.strategy == "freeze_all_except_last_layers":
            self._freeze_all_except_last_layers()
        else:
            raise ValueError(f"Unknown freeze strategy: {self.strategy}")
        
        # 验证冻结结果
        self._verify_freeze()
    
    def _freeze_layer_wise(self):
        """按层名称冻结指定层"""
        for layer_name in self.frozen_layers:
            if layer_name in self.layer_mapping:
                module = self.layer_mapping[layer_name]
                self._freeze_module(module, layer_name)
                self.logger(f"  ✅ Frozen layer: {layer_name}")
            else:
                self.logger(f"  ⚠️ Layer not found: {layer_name}")
    
    def _freeze_by_ratio(self):
        """按参数量比例冻结"""
        # 获取所有参数
        all_params = [(name, param) for name, param in self.model.named_parameters()]
        
        # 按参数在模型中的顺序排序（通常是从输入到输出）
        all_params.sort(key=lambda x: x[0])
        
        # 计算总参数量
        total_params = sum(p.numel() for _, p in all_params)
        freeze_threshold = int(total_params * self.freeze_ratio)
        
        self.logger(f"  Total parameters: {total_params:,}")
        self.logger(f"  Freezing first {self.freeze_ratio*100:.1f}% ({freeze_threshold:,} params)")
        
        # 冻结参数
        accumulated_params = 0
        for name, param in all_params:
            if accumulated_params < freeze_threshold:
                param.requires_grad = False
                accumulated_params += param.numel()
            else:
                param.requires_grad = True
        
        self.logger(f"  ✅ Frozen {accumulated_params:,} parameters ({accumulated_params/total_params:.1%})")
    
    def _freeze_all_except_head(self):
        """冻结除了head和fc之外的所有层"""
        for layer_name, module in self.layer_mapping.items():
            if layer_name != 'head':
                self._freeze_module(module, layer_name)
                self.logger(f"  ✅ Frozen layer: {layer_name}")
    
    def _freeze_all_except_last_layers(self):
        """冻结除了layer4和head之外的所有层（最适合头盔场景）"""
        layers_to_keep_trainable = ['layer4', 'head']
        
        for layer_name, module in self.layer_mapping.items():
            if layer_name not in layers_to_keep_trainable:
                self._freeze_module(module, layer_name)
                self.logger(f"  ✅ Frozen layer: {layer_name}")
            else:
                self.logger(f"  🎯 Keeping layer trainable: {layer_name}")
    
    def _freeze_module(self, module: torch.nn.Module, layer_name: str):
        """冻结整个模块及其子模块"""
        for name, param in module.named_parameters():
            param.requires_grad = False
    
    def _verify_freeze(self):
        """验证冻结结果"""
        total_params = 0
        frozen_params = 0
        layer_status = defaultdict(lambda: {'param_count': 0, 'frozen_count': 0})
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # 确定参数所属层
            layer_name = 'unknown'
            if 'conv1' in name or 'bn1' in name or 'stem_dw' in name:
                layer_name = 'stem'
            elif 'layer1' in name:
                layer_name = 'layer1'
            elif 'layer2' in name:
                layer_name = 'layer2'
            elif 'layer3' in name:
                layer_name = 'layer3'
            elif 'layer4' in name:
                layer_name = 'layer4'
            else:
                layer_name = 'head'
            
            layer_status[layer_name]['param_count'] += param_count
            if not param.requires_grad:
                frozen_params += param_count
                layer_status[layer_name]['frozen_count'] += param_count
        
        frozen_ratio = frozen_params / total_params if total_params > 0 else 0
        self.logger(f"📊 Freeze Verification:")
        self.logger(f"   Total parameters: {total_params:,}")
        self.logger(f"   Frozen parameters: {frozen_params:,} ({frozen_ratio:.1%})")
        self.logger(f"   Trainable parameters: {total_params - frozen_params:,} ({1 - frozen_ratio:.1%})")
        
        # 按层显示冻结状态
        self.logger("   Layer-wise status:")
        for layer_name, status in layer_status.items():
            layer_frozen_ratio = status['frozen_count'] / status['param_count'] if status['param_count'] > 0 else 0
            status_str = "🔒 FROZEN" if layer_frozen_ratio > 0.99 else "🎯 TRAINABLE"
            self.logger(f"      {layer_name}: {status_str} ({status['frozen_count']:,}/{status['param_count']:,} frozen)")
    
    def validate_pretrained_loading(self):
        """验证预训练权重是否正确加载"""
        self.logger("🔍 Validating pretrained weight loading...")
        
        mismatched_params = []
        for name, param in self.model.named_parameters():
            if name in self.initial_state:
                initial = self.initial_state[name]
                current = param.data.cpu()
                
                # 检查是否有变化（排除随机初始化的小差异）
                diff = torch.abs(initial - current).mean().item()
                if diff > 1e-6:  # 阈值，考虑浮点精度
                    mismatched_params.append((name, diff))
        
        if mismatched_params:
            self.logger(f"   ⚠️ {len(mismatched_params)} parameters changed after loading:")
            for name, diff in mismatched_params[:10]:  # 只显示前10个
                self.logger(f"      {name}: diff={diff:.6f}")
            if len(mismatched_params) > 10:
                self.logger(f"      ... and {len(mismatched_params)-10} more")
        else:
            self.logger("   ✅ All parameters match initial state - weights loaded correctly!")
    
    def get_freeze_info(self) -> Dict[str, Any]:
        """获取冻结信息用于日志和监控"""
        total_params = 0
        frozen_params = 0
        layer_status = {}
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # 确定参数所属层
            layer_name = 'unknown'
            if 'conv1' in name or 'bn1' in name or 'stem_dw' in name:
                layer_name = 'stem'
            elif 'layer1' in name:
                layer_name = 'layer1'
            elif 'layer2' in name:
                layer_name = 'layer2'
            elif 'layer3' in name:
                layer_name = 'layer3'
            elif 'layer4' in name:
                layer_name = 'layer4'
            else:
                layer_name = 'head'
            
            if layer_name not in layer_status:
                layer_status[layer_name] = {'param_count': 0, 'frozen_count': 0, 'frozen': False}
            
            layer_status[layer_name]['param_count'] += param_count
            if not param.requires_grad:
                frozen_params += param_count
                layer_status[layer_name]['frozen_count'] += param_count
        
        # 确定每层是否冻结
        for layer_name, status in layer_status.items():
            status['frozen'] = (status['frozen_count'] / status['param_count']) > 0.99
        
        frozen_ratio = frozen_params / total_params if total_params > 0 else 0
        trainable_ratio = 1 - frozen_ratio
        
        return {
            'frozen_ratio': frozen_ratio,
            'trainable_ratio': trainable_ratio,
            'frozen_count': frozen_params,
            'trainable_count': total_params - frozen_params,
            'total_count': total_params,
            'layer_status': layer_status,
            'strategy': self.strategy
        }


class GradientMonitor:
    """梯度监控器，用于监控冻结层的梯度流动"""
    
    def __init__(self, model: torch.nn.Module, logger):
        self.model = model
        self.logger = logger
        self.grad_stats = defaultdict(lambda: {'count': 0, 'grad_norm_sum': 0.0})
        self.last_log_step = 0
        self.config = {
            'enable_grad_monitor': True,
            'log_interval': 50,
            'detect_grad_leak': True
        }
    
    def monitor_gradients(self, current_step: int, tb_logger=None):
        """监控梯度，检测冻结层的梯度泄露"""
        if not self.config['enable_grad_monitor']:
            return
        
        has_grad_leak = False
        grad_leak_details = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                self.grad_stats[name]['count'] += 1
                self.grad_stats[name]['grad_norm_sum'] += grad_norm
                
                # 检测冻结层的梯度泄露
                if not param.requires_grad and grad_norm > 1e-8:  # 非零梯度
                    has_grad_leak = True
                    grad_leak_details.append((name, grad_norm))
        
        # 定期记录梯度统计
        if current_step - self.last_log_step >= self.config['log_interval']:
            self._log_gradient_stats(current_step, tb_logger)
            
            # 检测梯度泄露
            if has_grad_leak and self.config['detect_grad_leak']:
                self._log_grad_leak(current_step, grad_leak_details)
            
            self.last_log_step = current_step
    
    def _log_gradient_stats(self, step: int, tb_logger=None):
        """记录梯度统计信息"""
        self.logger(f"📈 Gradient stats at step {step}:")
        
        # 按层分组统计
        layer_stats = defaultdict(lambda: {'param_count': 0, 'grad_norm_sum': 0.0, 'grad_count': 0})
        
        for name, stats in self.grad_stats.items():
            if stats['count'] > 0:
                avg_grad_norm = stats['grad_norm_sum'] / stats['count']
                
                # 确定参数所属层
                layer_name = 'unknown'
                if 'conv1' in name or 'bn1' in name or 'stem_dw' in name:
                    layer_name = 'stem'
                elif 'layer1' in name:
                    layer_name = 'layer1'
                elif 'layer2' in name:
                    layer_name = 'layer2'
                elif 'layer3' in name:
                    layer_name = 'layer3'
                elif 'layer4' in name:
                    layer_name = 'layer4'
                else:
                    layer_name = 'head'
                
                layer_stats[layer_name]['grad_norm_sum'] += avg_grad_norm
                layer_stats[layer_name]['grad_count'] += 1
        
        # 记录每层平均梯度
        for layer_name, stats in layer_stats.items():
            if stats['grad_count'] > 0:
                avg_layer_grad = stats['grad_norm_sum'] / stats['grad_count']
                log_msg = f"   {layer_name}: avg_grad_norm={avg_layer_grad:.6f}"
                
                # 特别标记冻结层的梯度
                if 'stem' in layer_name or 'layer1' in layer_name or 'layer2' in layer_name or 'layer3' in layer_name:
                    if avg_layer_grad > 1e-8:
                        log_msg += " ⚠️(unexpected grad)"
                
                self.logger(log_msg)
                
                # TensorBoard记录
                if tb_logger is not None:
                    tb_logger.write(f"grad_norm/{layer_name}", avg_layer_grad, step)
        
        # 重置统计
        self.grad_stats.clear()
    
    def _log_grad_leak(self, step: int, leak_details: List[Tuple[str, float]]):
        """记录梯度泄露详情"""
        self.logger(f"🚨 GRADIENT LEAK DETECTED at step {step}!")
        self.logger(f"   {len(leak_details)} frozen parameters received non-zero gradients:")
        
        # 按梯度大小排序，显示最严重的
        leak_details.sort(key=lambda x: x[1], reverse=True)
        for name, grad_norm in leak_details[:10]:
            self.logger(f"      {name}: grad_norm={grad_norm:.6f}")
        
        if len(leak_details) > 10:
            self.logger(f"      ... and {len(leak_details)-10} more")
        
        # 建议修复措施
        self.logger("   💡 Suggested fixes:")
        self.logger("      1. Check if DDP find_unused_parameters=True is causing this")
        self.logger("      2. Verify that frozen parameters are not used in loss computation")
        self.logger("      3. Consider using torch.no_grad() context for frozen parts")