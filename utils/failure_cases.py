"""
失败案例网络定义模块
Failure Cases Network Definitions

提供常见的网络设计错误案例，用于教学演示
"""

import torch
import torch.nn as nn


class DeepMLPWithoutSkip(nn.Module):
    """
    100层普通MLP（无残差连接）
    
    问题：梯度消失
    现象：深层的梯度接近0，网络无法训练
    """
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=100):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, 2))
        
        self.activation = nn.Sigmoid()  # 容易梯度消失的激活函数
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # 输出层不加激活
        return x


class ConvToFullyConnected(nn.Module):
    """
    卷积层直接接超大全连接层
    
    问题：参数爆炸
    现象：内存占用巨大，训练速度慢
    """
    def __init__(self):
        super().__init__()
        # 简单的卷积层
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 问题所在：224x224的特征图直接flatten后接全连接
        # 参数量 = 64 * 224 * 224 * 1000 = 3,211,264,000 个参数！
        self.fc = nn.Linear(64 * 224 * 224, 1000)
    
    def forward(self, x):
        # 输入: [B, 3, 224, 224]
        x = self.relu(self.conv(x))  # [B, 64, 224, 224]
        x = x.view(x.size(0), -1)    # [B, 64*224*224]
        x = self.fc(x)               # [B, 1000]
        return x


class DeepNetWithoutNorm(nn.Module):
    """
    深度网络没有归一化层
    
    问题：训练不稳定，容易梯度爆炸或消失
    现象：Loss曲线剧烈震荡，难以收敛
    """
    def __init__(self, num_layers=20):
        super().__init__()
        layers = []
        
        # 输入层
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        
        # 深层卷积（无BatchNorm）
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TinyMLPWithHugeLR(nn.Module):
    """
    简单网络 + 超大学习率
    
    问题：梯度爆炸
    现象：Loss变成NaN，权重数值溢出
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_failure_case(case_name):
    """
    获取指定的失败案例
    
    Args:
        case_name: 案例名称
            - "deep_mlp": 100层MLP梯度消失
            - "conv_fc": 卷积接全连接参数爆炸
            - "no_norm": 无归一化训练不稳定
            - "huge_lr": 学习率过大梯度爆炸
    
    Returns:
        model: PyTorch模型
        description: 案例描述字典
    """
    cases = {
        "deep_mlp": {
            "model": DeepMLPWithoutSkip(),
            "name": "100层普通MLP（梯度消失）",
            "problem": "梯度消失",
            "symptom": "深层的梯度接近0，网络后层无法训练",
            "reason": "Sigmoid激活函数的导数最大值为0.25，100层连乘后梯度趋近于0",
            "solution": "使用残差连接(ResNet)或更换激活函数(ReLU/GELU)",
            "input_size": (1, 10),
            "loss_fn": "CrossEntropyLoss"
        },
        "conv_fc": {
            "model": ConvToFullyConnected(),
            "name": "卷积层直接接超大全连接",
            "problem": "参数爆炸",
            "symptom": "内存占用巨大（>12GB），训练速度极慢",
            "reason": "64×224×224的特征图flatten后有3,211,264个神经元，接1000分类需32亿参数",
            "solution": "使用全局平均池化(Global Average Pooling)降维",
            "input_size": (1, 3, 224, 224),
            "loss_fn": "CrossEntropyLoss"
        },
        "no_norm": {
            "model": DeepNetWithoutNorm(),
            "name": "20层卷积网络无归一化",
            "problem": "训练不稳定",
            "symptom": "Loss震荡，收敛困难，不同batch的激活值分布差异大",
            "reason": "深层网络的激活值分布会逐渐偏移，导致梯度不稳定",
            "solution": "在每层卷积后添加BatchNorm或LayerNorm",
            "input_size": (1, 3, 32, 32),
            "loss_fn": "CrossEntropyLoss"
        },
        "huge_lr": {
            "model": TinyMLPWithHugeLR(),
            "name": "简单MLP + 超大学习率",
            "problem": "梯度爆炸",
            "symptom": "Loss变成NaN，权重数值溢出到inf",
            "reason": "学习率过大（如lr=10.0），更新步长超出收敛范围",
            "solution": "使用合理的学习率（0.001-0.01），或使用学习率调度器",
            "input_size": (1, 10),
            "loss_fn": "CrossEntropyLoss",
            "bad_lr": 10.0,
            "good_lr": 0.01
        }
    }
    
    if case_name not in cases:
        raise ValueError(f"未知案例: {case_name}. 可用案例: {list(cases.keys())}")
    
    return cases[case_name]["model"], cases[case_name]


if __name__ == "__main__":
    # 测试案例
    print("=" * 60)
    print("失败案例测试")
    print("=" * 60)
    
    for case_name in ["deep_mlp", "conv_fc", "no_norm", "huge_lr"]:
        model, info = get_failure_case(case_name)
        print(f"\n案例: {info['name']}")
        print(f"问题: {info['problem']}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        # 测试前向传播
        try:
            x = torch.randn(info['input_size'])
            y = model(x)
            print(f"✅ 前向传播成功，输出形状: {y.shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
