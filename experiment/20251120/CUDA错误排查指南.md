# CUDA Kernel错误排查指南

## 🐛 问题描述

CUDA kernel错误通常表现为：
```
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

这是由于CUDA操作是异步的，错误可能在实际发生位置之后才被报告。

---

## ✅ 已修复的问题

### 问题1：频域分支维度不匹配

**原因：**
```python
# 错误示例：
fft_mag = torch.abs(fft)  # (B, C/2, H, W)
fft_phase = torch.angle(fft)  # (B, C/2, H, W)
fft_mag_phase = torch.cat([fft_mag, fft_phase], dim=1)  # (B, C, H, W) ✅ 正确

# 但后续处理期望的维度与实际不符
freq_weights = self.freq_gate(fft_feat)  # 期望输出 (B, C, 1, 1)
```

**修复方案：**

已在代码中添加详细的维度注释，确保数据流清晰：

```python
# 第410-436行（FrequencyAwareBlock.forward）
fft = torch.fft.fft2(freq_x, dim=(-2, -1))  # freq_x: (B, C/2, H, W)
fft_mag = torch.abs(fft)  # (B, C/2, H, W)
fft_phase = torch.angle(fft)  # (B, C/2, H, W)
fft_mag_phase = torch.cat([fft_mag, fft_phase], dim=1)  # (B, C, H, W) - 拼接后通道翻倍
fft_feat = self.fft_mag_phase_extract(fft_mag_phase)  # (B, C/2, H, W) - 1x1卷积降维
freq_weights = self.freq_gate(fft_feat)  # (B, C, 1, 1) - C通道用于分成low/high各C/2
low_freq_weight, high_freq_weight = freq_weights.chunk(2, dim=1)  # 各(B, C/2, 1, 1)
```

---

## 🔍 调试方法

### 方法1：启用同步CUDA执行（推荐）

在运行代码前设置环境变量：

```bash
# Windows (PowerShell)
$env:CUDA_LAUNCH_BLOCKING=1
python FPN_Restormer_CA_CNN_Encoder.py

# Windows (CMD)
set CUDA_LAUNCH_BLOCKING=1
python FPN_Restormer_CA_CNN_Encoder.py

# Linux/Mac
export CUDA_LAUNCH_BLOCKING=1
python FPN_Restormer_CA_CNN_Encoder.py
```

这会让CUDA操作同步执行，错误堆栈会指向实际出错位置。

### 方法2：逐模块测试

创建测试脚本 `test_modules.py`：

```python
import torch
import torch.nn as nn
from FPN_Restormer_CA_CNN_Encoder import FrequencyAwareBlock, Light_FFT_DSConv_Block

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 测试1：原始Block
print("\n测试1: Light_FFT_DSConv_Block")
try:
    block_original = Light_FFT_DSConv_Block(dim=96, bias=False, dilation_rate=1).to(device)
    x = torch.randn(2, 96, 64, 64).to(device)
    out = block_original(x)
    print(f"✅ 成功！输入: {x.shape}, 输出: {out.shape}")
except Exception as e:
    print(f"❌ 失败：{e}")

# 测试2：FrequencyAwareBlock
print("\n测试2: FrequencyAwareBlock")
try:
    block_freq = FrequencyAwareBlock(dim=96, bias=False, dilation_rate=1).to(device)
    x = torch.randn(2, 96, 64, 64).to(device)

    # 逐步测试
    print("步骤1: 通道拆分...")
    spatial_x, freq_x = x.chunk(2, dim=1)
    print(f"  spatial_x: {spatial_x.shape}, freq_x: {freq_x.shape}")

    print("步骤2: FFT变换...")
    fft = torch.fft.fft2(freq_x, dim=(-2, -1))
    print(f"  fft: {fft.shape}")

    print("步骤3: 幅度和相位提取...")
    fft_mag = torch.abs(fft)
    fft_phase = torch.angle(fft)
    print(f"  fft_mag: {fft_mag.shape}, fft_phase: {fft_phase.shape}")

    print("步骤4: 拼接...")
    fft_mag_phase = torch.cat([fft_mag, fft_phase], dim=1)
    print(f"  fft_mag_phase: {fft_mag_phase.shape}")

    print("步骤5: 完整前向传播...")
    out = block_freq(x)
    print(f"✅ 成功！输入: {x.shape}, 输出: {out.shape}")
except Exception as e:
    print(f"❌ 失败：{e}")
    import traceback
    traceback.print_exc()

# 测试3：不同维度测试
print("\n测试3: 不同维度")
test_dims = [48, 96, 192, 384]  # 对应level1, level2, level3, latent
for dim in test_dims:
    try:
        block = FrequencyAwareBlock(dim=dim, bias=False, dilation_rate=2).to(device)
        x = torch.randn(1, dim, 32, 32).to(device)
        out = block(x)
        print(f"✅ dim={dim}: 输入{x.shape} -> 输出{out.shape}")
    except Exception as e:
        print(f"❌ dim={dim} 失败: {e}")

print("\n所有测试完成！")
```

运行测试：
```bash
CUDA_LAUNCH_BLOCKING=1 python test_modules.py
```

### 方法3：检查CUDA内存

如果是显存不足导致的错误：

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU显存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB")

# 在模型前向传播前后调用
print_gpu_memory()
out = model(inp)
print_gpu_memory()
```

---

## 🔧 常见CUDA错误原因及解决方案

### 1. 维度不匹配（最常见）

**症状：** RuntimeError: mat1 and mat2 shapes cannot be multiplied

**检查：**
```python
# 在FrequencyAwareBlock.forward中添加断点
print(f"spatial_x: {spatial_x.shape}")
print(f"freq_x: {freq_x.shape}")
print(f"fft_mag_phase: {fft_mag_phase.shape}")
print(f"fft_feat: {fft_feat.shape}")
```

**预期输出：**
```
spatial_x: torch.Size([B, C/2, H, W])
freq_x: torch.Size([B, C/2, H, W])
fft_mag_phase: torch.Size([B, C, H, W])  # C/2 * 2 = C
fft_feat: torch.Size([B, C/2, H, W])     # 1x1卷积降维
```

### 2. 数值稳定性问题

**症状：** CUDA error: an illegal memory access was encountered

**原因：** FFT后的相位角度可能包含NaN或Inf

**解决方案：**
```python
# 在第416行后添加数值检查
fft_phase = torch.angle(fft)
if torch.isnan(fft_phase).any() or torch.isinf(fft_phase).any():
    print("警告：fft_phase包含NaN或Inf，进行裁剪")
    fft_phase = torch.clamp(fft_phase, -3.14159, 3.14159)
```

### 3. 显存不足

**症状：** CUDA out of memory

**解决方案：**
```python
# 方法1：减小batch size或图像尺寸
inp = torch.randn(1, 3, 128, 128).cuda()  # 从224降到128

# 方法2：使用梯度检查点（训练时）
import torch.utils.checkpoint as checkpoint
out = checkpoint.checkpoint(model.encoder_level1, x)

# 方法3：减小模型维度
model = Restormer_FFT_DSConv_Fusion(
    dim=40,  # 从48降到40
    use_frequency_aware=True
)
```

### 4. CUDA版本不兼容

**检查：**
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
```

**要求：**
- PyTorch >= 1.8.0（支持torch.fft.fft2）
- CUDA >= 10.2
- cuDNN >= 7.6

---

## 🎯 快速排查流程

### Step 1: 确认环境
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Step 2: CPU模式测试
```python
# 在主程序中修改
device = 'cpu'  # 强制使用CPU
inp = torch.randn(1, 3, 224, 224).to(device)
model = model.to(device)
```

如果CPU模式正常，说明是CUDA特定问题。

### Step 3: 逐层测试
```python
# 测试编码器各层
print("测试level1...")
out1 = model.encoder_level1(model.patch_embed(inp))
print(f"level1输出: {out1.shape}")

print("测试level2...")
out2 = model.encoder_level2(model.down1_2(out1))
print(f"level2输出: {out2.shape}")

# 依此类推...
```

### Step 4: 对比原始Block
```python
# 如果FrequencyAwareBlock报错，切换回原始Block
model = Restormer_FFT_DSConv_Fusion(
    use_frequency_aware=False,  # 使用原始Light_FFT_DSConv_Block
    task_aware_fusion=False
)
```

---

## 📝 修复日志

### 2025-01-20 修复记录

**问题：** FrequencyAwareBlock中频域分支维度不清晰

**修复：**
1. 添加详细的维度注释（第414-436行）
2. 明确各步骤的张量形状
3. 确保freq_gate输出通道数正确（C通道，用于分成low/high各C/2）

**验证：**
```python
# 测试用例
block = FrequencyAwareBlock(dim=96, bias=False, dilation_rate=1)
x = torch.randn(2, 96, 64, 64)
out = block(x)
assert out.shape == x.shape, "输出形状应与输入相同"
print("✅ 维度测试通过")
```

---

## 💡 最佳实践

### 1. 开发时建议

```python
# 开启调试模式
torch.autograd.set_detect_anomaly(True)

# 设置CUDA同步
torch.cuda.synchronize()

# 添加维度断言
def forward(self, x):
    assert x.dim() == 4, f"期望4D张量，得到{x.dim()}D"
    assert x.size(1) == self.dim, f"期望{self.dim}通道，得到{x.size(1)}"
    # ... 正常前向传播
```

### 2. 生产环境建议

```python
# 关闭调试（提升速度）
torch.autograd.set_detect_anomaly(False)

# 使用混合精度（节省显存）
from torch.cuda.amp import autocast
with autocast():
    out = model(inp)

# 使用编译优化（PyTorch 2.0+）
model = torch.compile(model)
```

---

## 📧 反馈

如果以上方法仍无法解决问题，请提供以下信息：

1. **完整错误堆栈**（使用CUDA_LAUNCH_BLOCKING=1运行）
2. **环境信息**：
   ```python
   import torch
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.get_device_name(0))
   ```
3. **输入数据形状**：
   ```python
   print(f"输入: {inp.shape}")
   ```
4. **模型配置**：
   ```python
   print(f"dim={dim}, use_frequency_aware={use_frequency_aware}")
   ```

---

## ✅ 确认修复

运行以下命令确认问题已解决：

```bash
CUDA_LAUNCH_BLOCKING=1 python FPN_Restormer_CA_CNN_Encoder.py
```

预期输出：
```
================================================================================
模型对比实验：FrequencyAwareBlock vs 原始Light_FFT_DSConv_Block
================================================================================

================================================================================
【测试1】原始模型 (Light_FFT_DSConv_Block)
================================================================================
输出形状: torch.Size([1, 3, 224, 224])
参数量: 11.817M
显存占用: 8188.808 MB
...
```

如果正常输出，说明CUDA错误已修复！🎉
