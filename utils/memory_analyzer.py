"""
内存分析器模块
Memory Analyzer Module

计算神经网络训练时的内存占用
包括：前向激活值、反向梯度、参数权重
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class LayerMemoryInfo:
    """单层内存信息"""
    def __init__(self, name: str, layer_type: str):
        self.name = name
        self.layer_type = layer_type
        
        # 前向传播
        self.input_memory = 0.0      # MB
        self.output_memory = 0.0     # MB
        self.param_memory = 0.0      # MB
        
        # 反向传播
        self.grad_memory = 0.0       # MB
        
        # 总计
        self.forward_peak = 0.0      # MB（前向峰值）
        self.backward_peak = 0.0     # MB（反向峰值）
        
        # 其他信息
        self.input_shape = None
        self.output_shape = None
        self.param_count = 0


def get_tensor_memory(shape: Tuple, dtype=torch.float32) -> float:
    """
    计算张量占用的内存（MB）
    
    Args:
        shape: 张量形状
        dtype: 数据类型
    
    Returns:
        memory: 内存占用（MB）
    """
    num_elements = np.prod(shape)
    
    # 不同数据类型的字节数
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
    }
    
    bytes_per_element = dtype_bytes.get(dtype, 4)
    total_bytes = num_elements * bytes_per_element
    memory_mb = total_bytes / (1024 ** 2)
    
    return memory_mb


def analyze_conv2d_memory(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    input_shape: Tuple[int, int, int, int],
    stride: int = 1,
    padding: int = 0
) -> LayerMemoryInfo:
    """
    分析Conv2d层的内存占用
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        input_shape: 输入形状 [B, C, H, W]
        stride: 步长
        padding: 填充
    
    Returns:
        info: 内存信息
    """
    info = LayerMemoryInfo("Conv2d", "Conv2d")
    
    B, C, H, W = input_shape
    
    # 计算输出形状
    H_out = (H + 2 * padding - kernel_size[0]) // stride + 1
    W_out = (W + 2 * padding - kernel_size[1]) // stride + 1
    output_shape = (B, out_channels, H_out, W_out)
    
    info.input_shape = input_shape
    info.output_shape = output_shape
    
    # 输入内存
    info.input_memory = get_tensor_memory(input_shape)
    
    # 输出内存
    info.output_memory = get_tensor_memory(output_shape)
    
    # 参数内存（权重 + 偏置）
    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
    bias_shape = (out_channels,)
    info.param_memory = get_tensor_memory(weight_shape) + get_tensor_memory(bias_shape)
    info.param_count = np.prod(weight_shape) + np.prod(bias_shape)
    
    # 梯度内存（与输出相同）
    info.grad_memory = info.output_memory
    
    # 峰值内存
    info.forward_peak = info.input_memory + info.output_memory + info.param_memory
    info.backward_peak = info.forward_peak + info.grad_memory
    
    return info


def analyze_linear_memory(
    in_features: int,
    out_features: int,
    input_shape: Tuple[int, int]
) -> LayerMemoryInfo:
    """
    分析Linear层的内存占用
    
    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        input_shape: 输入形状 [B, in_features]
    
    Returns:
        info: 内存信息
    """
    info = LayerMemoryInfo("Linear", "Linear")
    
    B = input_shape[0]
    output_shape = (B, out_features)
    
    info.input_shape = input_shape
    info.output_shape = output_shape
    
    # 输入内存
    info.input_memory = get_tensor_memory(input_shape)
    
    # 输出内存
    info.output_memory = get_tensor_memory(output_shape)
    
    # 参数内存（权重 + 偏置）
    weight_shape = (out_features, in_features)
    bias_shape = (out_features,)
    info.param_memory = get_tensor_memory(weight_shape) + get_tensor_memory(bias_shape)
    info.param_count = np.prod(weight_shape) + np.prod(bias_shape)
    
    # 梯度内存
    info.grad_memory = info.output_memory
    
    # 峰值内存
    info.forward_peak = info.input_memory + info.output_memory + info.param_memory
    info.backward_peak = info.forward_peak + info.grad_memory
    
    return info


def analyze_batchnorm_memory(
    num_features: int,
    input_shape: Tuple[int, ...]
) -> LayerMemoryInfo:
    """
    分析BatchNorm层的内存占用
    
    Args:
        num_features: 特征数量
        input_shape: 输入形状
    
    Returns:
        info: 内存信息
    """
    info = LayerMemoryInfo("BatchNorm", "BatchNorm")
    
    info.input_shape = input_shape
    info.output_shape = input_shape  # BatchNorm不改变形状
    
    # 输入内存
    info.input_memory = get_tensor_memory(input_shape)
    
    # 输出内存（与输入相同）
    info.output_memory = info.input_memory
    
    # 参数内存（gamma, beta, running_mean, running_var）
    param_shape = (num_features,)
    info.param_memory = get_tensor_memory(param_shape) * 4  # 4个参数
    info.param_count = num_features * 4
    
    # 梯度内存
    info.grad_memory = info.output_memory
    
    # 峰值内存
    info.forward_peak = info.input_memory + info.output_memory + info.param_memory
    info.backward_peak = info.forward_peak + info.grad_memory
    
    return info


def analyze_pooling_memory(
    input_shape: Tuple[int, int, int, int],
    kernel_size: int,
    stride: int
) -> LayerMemoryInfo:
    """
    分析Pooling层的内存占用
    
    Args:
        input_shape: 输入形状 [B, C, H, W]
        kernel_size: 池化核大小
        stride: 步长
    
    Returns:
        info: 内存信息
    """
    info = LayerMemoryInfo("Pooling", "Pooling")
    
    B, C, H, W = input_shape
    
    # 计算输出形状
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    output_shape = (B, C, H_out, W_out)
    
    info.input_shape = input_shape
    info.output_shape = output_shape
    
    # 输入内存
    info.input_memory = get_tensor_memory(input_shape)
    
    # 输出内存
    info.output_memory = get_tensor_memory(output_shape)
    
    # 参数内存（Pooling无参数）
    info.param_memory = 0.0
    info.param_count = 0
    
    # 梯度内存
    info.grad_memory = info.output_memory
    
    # 峰值内存
    info.forward_peak = info.input_memory + info.output_memory
    info.backward_peak = info.forward_peak + info.grad_memory
    
    return info


def analyze_model_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    detailed: bool = True
) -> Dict:
    """
    分析整个模型的内存占用
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        detailed: 是否返回详细信息
    
    Returns:
        result: 内存分析结果
    """
    model.eval()
    
    # 收集所有层的信息
    layers_info = []
    current_shape = input_shape
    
    # 简化版：只分析顺序模型的主要层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            info = analyze_conv2d_memory(
                module.in_channels,
                module.out_channels,
                (module.kernel_size[0], module.kernel_size[1]),
                current_shape,
                module.stride[0],
                module.padding[0]
            )
            info.name = name
            layers_info.append(info)
            current_shape = info.output_shape
            
        elif isinstance(module, nn.Linear):
            # 需要flatten
            if len(current_shape) > 2:
                flattened = (current_shape[0], np.prod(current_shape[1:]))
                current_shape = flattened
            
            info = analyze_linear_memory(
                module.in_features,
                module.out_features,
                current_shape
            )
            info.name = name
            layers_info.append(info)
            current_shape = info.output_shape
            
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            info = analyze_batchnorm_memory(
                module.num_features,
                current_shape
            )
            info.name = name
            layers_info.append(info)
    
    # 计算总计
    total_forward_memory = sum(layer.output_memory for layer in layers_info)
    total_backward_memory = sum(layer.grad_memory for layer in layers_info)
    total_param_memory = sum(layer.param_memory for layer in layers_info)
    peak_memory = max(layer.backward_peak for layer in layers_info) if layers_info else 0
    
    # 找出内存瓶颈层
    if layers_info:
        bottleneck_layer = max(layers_info, key=lambda x: x.backward_peak)
        bottleneck_percentage = (bottleneck_layer.backward_peak / peak_memory * 100) if peak_memory > 0 else 0
    else:
        bottleneck_layer = None
        bottleneck_percentage = 0
    
    result = {
        "layers": layers_info if detailed else None,
        "summary": {
            "total_forward_memory": total_forward_memory,
            "total_backward_memory": total_backward_memory,
            "total_param_memory": total_param_memory,
            "peak_memory": peak_memory,
            "num_layers": len(layers_info)
        },
        "bottleneck": {
            "layer": bottleneck_layer,
            "percentage": bottleneck_percentage
        }
    }
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("内存分析器测试")
    print("=" * 60)
    
    # 测试Conv2d
    print("\n### Conv2d内存分析 ###")
    conv_info = analyze_conv2d_memory(
        in_channels=3,
        out_channels=64,
        kernel_size=(7, 7),
        input_shape=(1, 3, 224, 224),
        stride=2,
        padding=3
    )
    print(f"输入形状: {conv_info.input_shape}")
    print(f"输出形状: {conv_info.output_shape}")
    print(f"输入内存: {conv_info.input_memory:.2f} MB")
    print(f"输出内存: {conv_info.output_memory:.2f} MB")
    print(f"参数内存: {conv_info.param_memory:.2f} MB")
    print(f"梯度内存: {conv_info.grad_memory:.2f} MB")
    print(f"前向峰值: {conv_info.forward_peak:.2f} MB")
    print(f"反向峰值: {conv_info.backward_peak:.2f} MB")
    
    # 测试Linear
    print("\n### Linear内存分析 ###")
    linear_info = analyze_linear_memory(
        in_features=512,
        out_features=1000,
        input_shape=(1, 512)
    )
    print(f"输入形状: {linear_info.input_shape}")
    print(f"输出形状: {linear_info.output_shape}")
    print(f"输入内存: {linear_info.input_memory:.4f} MB")
    print(f"输出内存: {linear_info.output_memory:.4f} MB")
    print(f"参数内存: {linear_info.param_memory:.2f} MB")
    print(f"参数数量: {linear_info.param_count:,}")
    
    # 测试简单模型
    print("\n### 简单模型内存分析 ###")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.fc = nn.Linear(128 * 56 * 56, 1000)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    result = analyze_model_memory(model, (1, 3, 224, 224))
    
    print(f"\n总前向内存: {result['summary']['total_forward_memory']:.2f} MB")
    print(f"总反向内存: {result['summary']['total_backward_memory']:.2f} MB")
    print(f"总参数内存: {result['summary']['total_param_memory']:.2f} MB")
    print(f"峰值内存: {result['summary']['peak_memory']:.2f} MB")
    print(f"分析层数: {result['summary']['num_layers']}")
    
    if result['bottleneck']['layer']:
        print(f"\n内存瓶颈层: {result['bottleneck']['layer'].name}")
        print(f"占比: {result['bottleneck']['percentage']:.1f}%")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
