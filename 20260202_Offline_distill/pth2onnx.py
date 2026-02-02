import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
from resnet_sigma_inf import InfNet

def load_model_and_weights(checkpoint_path):
    print("正在加载模型...")
    model = InfNet()
    
    if os.path.exists(checkpoint_path):
        print(f"正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            elif key.startswith('model.'):
                new_key = key[6:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功!")
    else:
        print(f"警告: 权重文件不存在 {checkpoint_path}")
        return None
    
    model.eval()
    return model

def convert_to_onnx(model, input_shape, output_path, opset_version=13):
    print(f"正在转换为ONNX格式...")
    print(f"输入形状: {input_shape}")
    print(f"输出路径: {output_path}")
    
    dummy_input = torch.randn(input_shape)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        print(f"ONNX模型已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return False

def verify_onnx_model(onnx_path):
    print("正在验证ONNX模型...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过!")
        return True
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        return False

def compare_outputs(pytorch_model, onnx_path, input_shape, num_tests=10):
    print(f"正在比较精度，测试样本数: {num_tests}")
    
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print("ONNX推理会话创建成功")
    except Exception as e:
        print(f"创建ONNX推理会话失败: {e}")
        return None, None
    
    max_diff = 0
    all_diffs = []
    
    with torch.no_grad():
        for i in range(num_tests):
            test_input = torch.randn(input_shape)
            pytorch_output = pytorch_model(test_input)
            pytorch_output_np = pytorch_output.cpu().numpy()
            
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            diff = np.abs(pytorch_output_np - onnx_output)
            max_diff_current = np.max(diff)
            mean_diff_current = np.mean(diff)
            
            max_diff = max(max_diff, max_diff_current)
            all_diffs.append(mean_diff_current)
            
            print(f"测试 {i+1:2d}: 最大差异={max_diff_current:.8f}, 平均差异={mean_diff_current:.8f}")
    
    mean_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)
    
    print("="*60)
    print("精度比较结果:")
    print(f"最大绝对差异: {max_diff:.8f}")
    print(f"平均绝对差异: {mean_diff:.8f}")
    print(f"差异标准差: {std_diff:.8f}")
    
    if max_diff < 1e-5:
        print("精度验证通过! 差异很小，转换成功")
    elif max_diff < 1e-3:
        print("精度验证基本通过，有轻微差异")
    else:
        print("精度验证失败，差异较大")
    
    return max_diff, mean_diff

def get_model_info(pytorch_model, onnx_path):
    print("="*60)
    print("模型信息:")
    
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    print(f"PyTorch模型参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    if os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX模型文件大小: {onnx_size:.2f} MB")
        try:
            onnx_model = onnx.load(onnx_path)
            print(f"ONNX模型输入: {[(input.name, input.type.tensor_type.shape) for input in onnx_model.graph.input]}")
            print(f"ONNX模型输出: {[(output.name, output.type.tensor_type.shape) for output in onnx_model.graph.output]}")
        except Exception as e:
            print(f"读取ONNX模型信息失败: {e}")

def main():
    checkpoint_path = "/data/jg-face-unlock-fea-impove/vivo_new/helmet/txy_light_helmet_990_10000/pth/checkpoint_428.pt"
    output_onnx_path = "/data/jg-face-unlock-fea-impove/vivo_new/helmet/txy_light_helmet_990_10000/pth/checkpoint_428.onnx"
    input_shape = (1, 3, 128, 128)
    opset_version = 13
    
    print("="*60)
    print("PyTorch模型转ONNX脚本")
    print("="*60)
    
    # 1. 加载模型
    model = load_model_and_weights(checkpoint_path)
    if model is None:
        return
    
    # 2. 转换ONNX
    if not convert_to_onnx(model, input_shape, output_onnx_path, opset_version):
        return
    
    # 3. 验证ONNX
    if not verify_onnx_model(output_onnx_path):
        return
    
    # 4. 精度比较
    max_diff, mean_diff = compare_outputs(model, output_onnx_path, input_shape, num_tests=10)
    if max_diff is None:
        return
    
    # 5. 模型信息
    get_model_info(model, output_onnx_path)
    
    print("="*60)
    print("转换完成!")
    print(f"ONNX模型保存在: {output_onnx_path}")
    print(f"最大精度差异: {max_diff:.8f}")
    print(f"平均精度差异: {mean_diff:.8f}")
    print("="*60)

if __name__ == "__main__":
    main()
