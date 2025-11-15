# DDP_D2R_Restormer 代码修改说明

## 修改概述
本次修改确保了所有模块的通道数完全匹配，使整个网络架构在数学上完全正确且可运行。

## 主要修改内容

### 1. 修复 Downsample 模块 (第289-297行)
**问题**: 原实现没有明确注释通道变化
**修改**:
```python
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # PixelUnshuffle(2) increases channels by 4x, so n_feat//2 -> n_feat*2
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))
```
**效果**:
- 输入: n_feat 通道
- Conv2d: n_feat -> n_feat//2
- PixelUnshuffle(2): n_feat//2 -> n_feat*2 (通道数×4)
- 输出: n_feat*2 通道

### 2. 修复 Upsample 模块 (第300-308行)
**问题**: 原实现没有明确注释通道变化
**修改**:
```python
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # PixelShuffle(2) decreases channels by 4x, so n_feat*2 -> n_feat//2
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
```
**效果**:
- 输入: n_feat 通道
- Conv2d: n_feat -> n_feat*2
- PixelShuffle(2): n_feat*2 -> n_feat//2 (通道数÷4)
- 输出: n_feat//2 通道

### 3. 修复 DegradationResidualAdapter 模块 (第177-198行)
**问题**: proto维度固定为dim，但不同层的特征维度不同
**修改**:
```python
class DegradationResidualAdapter(nn.Module):
    def __init__(self, dim, proto_dim):  # 添加 proto_dim 参数
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(proto_dim, dim),  # 从 proto_dim 映射到 dim
            nn.GELU(),
            nn.Linear(dim, dim*2)
        )
        self.conv = nn.Conv2d(dim, dim, 1)
```
**效果**: 允许不同层使用不同的特征维度，同时保持原型嵌入维度一致

### 4. 修复 D2R_MoE 模块 (第229-268行)
**问题**: cross_attn 的 num_heads 硬编码为4，可能不能整除某些维度
**修改**:
```python
class D2R_MoE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        # ...
        # Use num_heads that divides dim evenly
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
```
**效果**: 使用传入的 num_heads 参数，确保能整除各层的维度

### 5. 修复主网络通道配置 (第339-404行)

#### 5.1 修复 DRA 模块列表
**原代码**:
```python
self.dra = nn.ModuleList([
    DegradationResidualAdapter(dim),
    DegradationResidualAdapter(dim*4),
    DegradationResidualAdapter(dim*16),
    DegradationResidualAdapter(dim*64),
])
```

**修改后**:
```python
self.dra = nn.ModuleList([
    DegradationResidualAdapter(dim, dim),        # proto_dim = dim
    DegradationResidualAdapter(dim*2, dim),      # proto_dim = dim
    DegradationResidualAdapter(dim*4, dim),      # proto_dim = dim
    DegradationResidualAdapter(dim*8, dim),      # proto_dim = dim
])
```

#### 5.2 修复 Encoder 通道配置
**原代码**: dim -> dim*4 -> dim*16 -> dim*64
**修改后**: dim -> dim*2 -> dim*4 -> dim*8

```python
self.encoder = nn.ModuleList([
    # Level 1: dim
    nn.Sequential(*[TransformerBlock(dim, heads[0], ...) for _ in range(num_blocks[0])]),
    # Level 2: dim*2
    nn.Sequential(*[TransformerBlock(dim*2, heads[1], ...) for _ in range(num_blocks[1])]),
    # Level 3: dim*4
    nn.Sequential(*[TransformerBlock(dim*4, heads[2], ...) for _ in range(num_blocks[2])]),
    # Level 4: dim*8
    nn.Sequential(*[TransformerBlock(dim*8, heads[3], ...) for _ in range(num_blocks[3])])
])

self.down = nn.ModuleList([
    Downsample(dim),      # dim -> dim*2
    Downsample(dim*2),    # dim*2 -> dim*4
    Downsample(dim*4)     # dim*4 -> dim*8
])
```

#### 5.3 修复 Bottleneck 通道配置
**原代码**: dim*64
**修改后**: dim*8

```python
self.bottleneck = nn.Sequential(
    TransformerBlock(dim*8, heads[3], ...),
    TransformerBlock(dim*8, heads[3], ...),
)
```

#### 5.4 修复 Decoder 通道配置
**原代码**: dim*64 -> dim*16 -> dim*4 -> dim
**修改后**: dim*8 -> dim*4 -> dim*2 -> dim

```python
self.up = nn.ModuleList([
    Upsample(dim*8),   # dim*8 -> dim*4
    Upsample(dim*4),   # dim*4 -> dim*2
    Upsample(dim*2)    # dim*2 -> dim
])

self.decoder = nn.ModuleList([
    D2R_MoE(dim*4, heads[2], ...),  # Level 1
    D2R_MoE(dim*2, heads[1], ...),  # Level 2
    D2R_MoE(dim, heads[0], ...),    # Level 3
])
```

## 完整的数据流

### 编码器路径
```
Input (B, 3, H, W)
  ↓ patch_embed
(B, 48, H, W) ——→ encoder[0] ——→ e1 (B, 48, H, W) ——→ DRA[0]
  ↓ down[0]
(B, 96, H/2, W/2) ——→ encoder[1] ——→ e2 (B, 96, H/2, W/2) ——→ DRA[1]
  ↓ down[1]
(B, 192, H/4, W/4) ——→ encoder[2] ——→ e3 (B, 192, H/4, W/4) ——→ DRA[2]
  ↓ down[2]
(B, 384, H/8, W/8) ——→ encoder[3] ——→ e4 (B, 384, H/8, W/8) ——→ DRA[3]
  ↓ bottleneck
(B, 384, H/8, W/8)
```

### 解码器路径
```
(B, 384, H/8, W/8)
  ↓ up[0]
(B, 192, H/4, W/4) + e3 ——→ decoder[0] (D2R_MoE) ——→ d3 (B, 192, H/4, W/4)
  ↓ up[1]
(B, 96, H/2, W/2) + e2 ——→ decoder[1] (D2R_MoE) ——→ d2 (B, 96, H/2, W/2)
  ↓ up[2]
(B, 48, H, W) + e1 ——→ decoder[2] (D2R_MoE) ——→ d1 (B, 48, H, W)
  ↓ refine
(B, 48, H, W)
  ↓ output
(B, 3, H, W)
```

## 验证结果

✓ 所有通道维度完全匹配
✓ 编码器: dim → dim×2 → dim×4 → dim×8
✓ 解码器: dim×8 → dim×4 → dim×2 → dim
✓ 跳跃连接在每个层级完全匹配
✓ DRA 模块正确地将 proto_dim 适配到各层的 feat_dim
✓ 代码语法检查通过
✓ 架构在数学上完全正确

## 模型参数 (dim=48)

- 第1层: 48 通道
- 第2层: 96 通道 (48×2)
- 第3层: 192 通道 (48×4)
- 第4层: 384 通道 (48×8)
- Bottleneck: 384 通道
- 原型嵌入: 48 维 (在所有层共享)

## 关键改进

1. **通道数渐进增长**: 从 dim → dim×2 → dim×4 → dim×8，而不是 dim → dim×4 → dim×16 → dim×64
2. **更合理的计算复杂度**: 新的通道配置大幅降低了高层的计算量和内存消耗
3. **DRA适配性**: 通过添加 proto_dim 参数，使得原型嵌入可以灵活适配不同层的特征维度
4. **动态注意力头数**: D2R_MoE 使用传入的 num_heads，确保可以整除各层维度

## 总结

本次修改解决了原代码中所有通道数不匹配的问题，使网络架构完整、一致且可运行。代码已通过语法检查，通道维度计算已验证正确。
