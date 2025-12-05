"""
归一化层实现和对比模块
Normalization Layers Comparison

实现和对比BatchNorm、LayerNorm、GroupNorm的工作机制
"""

import torch
import torch.nn as nn
import numpy as np


def apply_batch_norm(x, eps=1e-5):
    """
    手动实现BatchNorm（用于教学演示）
    
    对于输入 x: [Batch, Channel, Height, Width]
    在 (Batch, Height, Width) 维度上计算均值和方差
    
    Args:
        x: 输入张量 [B, C, H, W]
        eps: 数值稳定项
    
    Returns:
        normalized: 归一化后的张量
        stats: 统计信息字典
    """
    # 计算每个通道的均值和方差（在B, H, W维度上）
    # x.shape = [B, C, H, W]
    mean = x.mean(dim=[0, 2, 3], keepdim=True)  # [1, C, 1, 1]
    var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)  # [1, C, 1, 1]
    
    # 归一化
    normalized = (x - mean) / torch.sqrt(var + eps)
    
    stats = {
        "mean": mean.squeeze().detach().cpu().numpy(),
        "var": var.squeeze().detach().cpu().numpy(),
        "std": torch.sqrt(var).squeeze().detach().cpu().numpy(),
        "normalized_mean": normalized.mean().item(),
        "normalized_std": normalized.std().item()
    }
    
    return normalized, stats


def apply_layer_norm(x, eps=1e-5):
    """
    手动实现LayerNorm（用于教学演示）
    
    对于输入 x: [Batch, Channel, Height, Width]
    在 (Channel, Height, Width) 维度上计算均值和方差
    
    Args:
        x: 输入张量 [B, C, H, W]
        eps: 数值稳定项
    
    Returns:
        normalized: 归一化后的张量
        stats: 统计信息字典
    """
    # 计算每个样本的均值和方差（在C, H, W维度上）
    mean = x.mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
    var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)  # [B, 1, 1, 1]
    
    # 归一化
    normalized = (x - mean) / torch.sqrt(var + eps)
    
    stats = {
        "mean": mean.squeeze().detach().cpu().numpy(),
        "var": var.squeeze().detach().cpu().numpy(),
        "std": torch.sqrt(var).squeeze().detach().cpu().numpy(),
        "normalized_mean": normalized.mean().item(),
        "normalized_std": normalized.std().item()
    }
    
    return normalized, stats


def apply_group_norm(x, num_groups=32, eps=1e-5):
    """
    手动实现GroupNorm（用于教学演示）
    
    对于输入 x: [Batch, Channel, Height, Width]
    将Channel分成num_groups组，在每组内归一化
    
    Args:
        x: 输入张量 [B, C, H, W]
        num_groups: 分组数量
        eps: 数值稳定项
    
    Returns:
        normalized: 归一化后的张量
        stats: 统计信息字典
    """
    B, C, H, W = x.shape
    
    # 确保通道数能被组数整除
    if C % num_groups != 0:
        num_groups = C  # 退化为LayerNorm
    
    # 重塑为 [B, num_groups, C//num_groups, H, W]
    x_grouped = x.view(B, num_groups, C // num_groups, H, W)
    
    # 在每组内计算均值和方差
    mean = x_grouped.mean(dim=[2, 3, 4], keepdim=True)  # [B, num_groups, 1, 1, 1]
    var = x_grouped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
    
    # 归一化
    normalized = (x_grouped - mean) / torch.sqrt(var + eps)
    
    # 恢复原始形状
    normalized = normalized.view(B, C, H, W)
    
    stats = {
        "num_groups": num_groups,
        "mean": mean.squeeze().detach().cpu().numpy(),
        "var": var.squeeze().detach().cpu().numpy(),
        "std": torch.sqrt(var).squeeze().detach().cpu().numpy(),
        "normalized_mean": normalized.mean().item(),
        "normalized_std": normalized.std().item()
    }
    
    return normalized, stats


def compare_normalization_methods(x, num_groups=32):
    """
    对比三种归一化方法
    
    Args:
        x: 输入张量 [B, C, H, W]
        num_groups: GroupNorm的分组数
    
    Returns:
        results: 包含三种方法结果的字典
    """
    results = {}
    
    # 原始数据统计
    results["original"] = {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item()
    }
    
    # BatchNorm
    bn_normalized, bn_stats = apply_batch_norm(x.clone())
    results["batch_norm"] = {
        "normalized": bn_normalized,
        "stats": bn_stats
    }
    
    # LayerNorm
    ln_normalized, ln_stats = apply_layer_norm(x.clone())
    results["layer_norm"] = {
        "normalized": ln_normalized,
        "stats": ln_stats
    }
    
    # GroupNorm
    gn_normalized, gn_stats = apply_group_norm(x.clone(), num_groups)
    results["group_norm"] = {
        "normalized": gn_normalized,
        "stats": gn_stats
    }
    
    return results


class SimpleCNNWithNorm(nn.Module):
    """
    简单CNN，可选择不同的归一化层
    用于对比不同归一化方法的效果
    """
    def __init__(self, norm_type="batch", num_groups=32):
        """
        Args:
            norm_type: "batch", "layer", "group", "none"
            num_groups: GroupNorm的分组数
        """
        super().__init__()
        self.norm_type = norm_type
        
        # Conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif norm_type == "layer":
            self.norm1 = nn.LayerNorm([64, 32, 32])  # 需要指定形状
        elif norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups, 64)
        else:
            self.norm1 = nn.Identity()
        
        # Conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        if norm_type == "batch":
            self.norm2 = nn.BatchNorm2d(128)
        elif norm_type == "layer":
            self.norm2 = nn.LayerNorm([128, 32, 32])
        elif norm_type == "group":
            self.norm2 = nn.GroupNorm(num_groups, 128)
        else:
            self.norm2 = nn.Identity()
        
        # Conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        if norm_type == "batch":
            self.norm3 = nn.BatchNorm2d(256)
        elif norm_type == "layer":
            self.norm3 = nn.LayerNorm([256, 32, 32])
        elif norm_type == "group":
            self.norm3 = nn.GroupNorm(num_groups, 256)
        else:
            self.norm3 = nn.Identity()
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x, return_activations=False):
        activations = []
        
        # Layer 1
        x = self.conv1(x)
        if return_activations:
            activations.append(("conv1_before_norm", x.clone()))
        x = self.norm1(x)
        if return_activations:
            activations.append(("conv1_after_norm", x.clone()))
        x = self.relu(x)
        
        # Layer 2
        x = self.conv2(x)
        if return_activations:
            activations.append(("conv2_before_norm", x.clone()))
        x = self.norm2(x)
        if return_activations:
            activations.append(("conv2_after_norm", x.clone()))
        x = self.relu(x)
        
        # Layer 3
        x = self.conv3(x)
        if return_activations:
            activations.append(("conv3_before_norm", x.clone()))
        x = self.norm3(x)
        if return_activations:
            activations.append(("conv3_after_norm", x.clone()))
        x = self.relu(x)
        
        # Classification head
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if return_activations:
            return x, activations
        return x


def get_normalization_comparison_info():
    """
    获取三种归一化方法的对比信息
    
    Returns:
        info: 对比信息字典
    """
    info = {
        "batch_norm": {
            "name": "Batch Normalization",
            "formula": "y = (x - μ_B) / √(σ²_B + ε)",
            "normalization_dim": "在 (Batch, Height, Width) 维度归一化",
            "when_to_use": [
                "卷积神经网络（CNN）",
                "大batch size训练（>16）",
                "数据分布相对稳定",
                "需要快速训练和收敛"
            ],
            "advantages": [
                "加速训练收敛",
                "允许使用更大学习率",
                "减少对初始化的依赖",
                "有轻微正则化效果"
            ],
            "disadvantages": [
                "依赖batch size（小batch效果差）",
                "训练和推理行为不一致",
                "不适合序列任务（RNN/LSTM）",
                "不适合在线学习"
            ]
        },
        "layer_norm": {
            "name": "Layer Normalization",
            "formula": "y = (x - μ_L) / √(σ²_L + ε)",
            "normalization_dim": "在 (Channel, Height, Width) 维度归一化",
            "when_to_use": [
                "Transformer架构",
                "循环神经网络（RNN/LSTM）",
                "小batch size场景",
                "序列长度可变的任务"
            ],
            "advantages": [
                "与batch size无关",
                "训练和推理行为一致",
                "适合序列任务",
                "适合在线学习"
            ],
            "disadvantages": [
                "在CNN上效果不如BatchNorm",
                "计算量略大",
                "对于某些任务可能过于强的约束"
            ]
        },
        "group_norm": {
            "name": "Group Normalization",
            "formula": "y = (x - μ_G) / √(σ²_G + ε)",
            "normalization_dim": "将通道分组，在每组内归一化",
            "when_to_use": [
                "小batch size训练（batch=1, 2）",
                "目标检测/实例分割",
                "视频处理（时空数据）",
                "BatchNorm效果不佳的场景"
            ],
            "advantages": [
                "与batch size无关",
                "在小batch场景效果好",
                "训练和推理行为一致",
                "介于BatchNorm和LayerNorm之间"
            ],
            "disadvantages": [
                "需要调整组数（hyperparameter）",
                "在大batch下不如BatchNorm",
                "计算略复杂"
            ]
        }
    }
    
    return info


if __name__ == "__main__":
    print("=" * 60)
    print("归一化层对比测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    channels = 64
    height = 32
    width = 32
    
    # 模拟不均匀的激活值分布
    x = torch.randn(batch_size, channels, height, width) * 10 + 5
    
    print(f"\n输入形状: {x.shape}")
    print(f"输入均值: {x.mean():.4f}")
    print(f"输入标准差: {x.std():.4f}")
    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
    
    # 对比三种归一化方法
    print("\n" + "=" * 60)
    print("归一化对比")
    print("=" * 60)
    
    results = compare_normalization_methods(x, num_groups=32)
    
    for method in ["batch_norm", "layer_norm", "group_norm"]:
        print(f"\n### {method.upper()} ###")
        stats = results[method]["stats"]
        print(f"归一化后均值: {stats['normalized_mean']:.6f}")
        print(f"归一化后标准差: {stats['normalized_std']:.6f}")
    
    # 测试CNN模型
    print("\n" + "=" * 60)
    print("SimpleCNN模型测试")
    print("=" * 60)
    
    x_input = torch.randn(2, 3, 32, 32)
    
    for norm_type in ["batch", "layer", "group", "none"]:
        print(f"\n### {norm_type.upper()} ###")
        model = SimpleCNNWithNorm(norm_type=norm_type, num_groups=32)
        
        try:
            output = model(x_input)
            print(f"✅ 前向传播成功，输出形状: {output.shape}")
            
            # 统计参数量
            params = sum(p.numel() for p in model.parameters())
            print(f"参数量: {params:,}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
