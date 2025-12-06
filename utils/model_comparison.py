"""
模型对比工具模块
Model Comparison Utilities

提供模型对比、训练曲线生成等功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


def get_model_info(model_name: str) -> Dict:
    """
    获取模型的基本信息
    
    Args:
        model_name: 模型名称
    
    Returns:
        info: 模型信息字典
    """
    model_configs = {
        "ResNet-18": {
            "type": "CNN",
            "params": 11.7,  # Million
            "flops": 1.8,    # GFLOPs
            "depth": 18,
            "architecture": "残差网络",
            "inductive_bias": "强（平移不变性、局部性）",
            "best_for": "小到中等数据集、需要快速训练",
            "pretrain_dataset": "ImageNet-1K",
            "year": 2015
        },
        "ResNet-50": {
            "type": "CNN",
            "params": 25.6,
            "flops": 4.1,
            "depth": 50,
            "architecture": "残差网络",
            "inductive_bias": "强（平移不变性、局部性）",
            "best_for": "中等到大型数据集",
            "pretrain_dataset": "ImageNet-1K",
            "year": 2015
        },
        "MobileNet-V2": {
            "type": "CNN",
            "params": 3.5,
            "flops": 0.3,
            "depth": 53,
            "architecture": "深度可分离卷积",
            "inductive_bias": "强（平移不变性、局部性）",
            "best_for": "边缘设备、资源受限场景",
            "pretrain_dataset": "ImageNet-1K",
            "year": 2018
        },
        "ViT-Tiny": {
            "type": "Transformer",
            "params": 5.7,
            "flops": 1.3,
            "depth": 12,
            "architecture": "Vision Transformer",
            "inductive_bias": "弱（需要从数据学习）",
            "best_for": "大数据集、需要全局建模",
            "pretrain_dataset": "ImageNet-21K",
            "year": 2020
        },
        "ViT-Small": {
            "type": "Transformer",
            "params": 22.0,
            "flops": 4.6,
            "depth": 12,
            "architecture": "Vision Transformer",
            "inductive_bias": "弱（需要从数据学习）",
            "best_for": "大数据集、需要全局建模",
            "pretrain_dataset": "ImageNet-21K",
            "year": 2020
        },
        "ViT-Base": {
            "type": "Transformer",
            "params": 86.0,
            "flops": 17.6,
            "depth": 12,
            "architecture": "Vision Transformer",
            "inductive_bias": "弱（需要从数据学习）",
            "best_for": "超大数据集、云端部署",
            "pretrain_dataset": "ImageNet-21K",
            "year": 2020
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"未知模型: {model_name}")
    
    return model_configs[model_name]


def generate_training_curve(
    model_type: str,
    dataset_size: str = "small",
    num_epochs: int = 100,
    seed: int = 42
) -> Dict:
    """
    生成模拟的训练曲线
    
    基于真实的训练规律，生成合理的训练曲线数据
    用于快速演示，避免实际训练
    
    Args:
        model_type: "CNN" 或 "Transformer"
        dataset_size: "small" (10K), "medium" (50K), "large" (500K)
        num_epochs: 训练轮数
        seed: 随机种子
    
    Returns:
        curves: 包含训练曲线的字典
    """
    np.random.seed(seed)
    
    # 根据模型类型和数据集大小设置参数
    if model_type == "CNN":
        if dataset_size == "small":
            initial_loss = 2.3
            final_loss = 0.3
            convergence_speed = 0.15
            initial_acc = 0.10
            final_acc = 0.85
            acc_convergence_speed = 0.12
        elif dataset_size == "medium":
            initial_loss = 2.3
            final_loss = 0.25
            convergence_speed = 0.12
            initial_acc = 0.10
            final_acc = 0.88
            acc_convergence_speed = 0.10
        else:  # large
            initial_loss = 2.3
            final_loss = 0.20
            convergence_speed = 0.10
            initial_acc = 0.10
            final_acc = 0.91
            acc_convergence_speed = 0.08
    else:  # Transformer
        if dataset_size == "small":
            initial_loss = 2.3
            final_loss = 0.5
            convergence_speed = 0.08
            initial_acc = 0.10
            final_acc = 0.75  # 小数据集上ViT表现差
            acc_convergence_speed = 0.06
        elif dataset_size == "medium":
            initial_loss = 2.3
            final_loss = 0.28
            convergence_speed = 0.10
            initial_acc = 0.10
            final_acc = 0.87
            acc_convergence_speed = 0.09
        else:  # large
            initial_loss = 2.3
            final_loss = 0.15
            convergence_speed = 0.12
            initial_acc = 0.10
            final_acc = 0.94  # 大数据集上ViT超过CNN
            acc_convergence_speed = 0.10
    
    # 生成Loss曲线（指数衰减 + 噪声）
    epochs = np.arange(num_epochs)
    train_loss = final_loss + (initial_loss - final_loss) * np.exp(-convergence_speed * epochs)
    train_loss += np.random.normal(0, 0.05, num_epochs)  # 添加噪声
    train_loss = np.clip(train_loss, 0, None)
    
    # 验证集Loss（略高于训练集）
    val_loss = train_loss * 1.1 + np.random.normal(0, 0.03, num_epochs)
    val_loss = np.clip(val_loss, 0, None)
    
    # 生成Accuracy曲线
    train_acc = final_acc - (final_acc - initial_acc) * np.exp(-acc_convergence_speed * epochs)
    train_acc += np.random.normal(0, 0.02, num_epochs)
    train_acc = np.clip(train_acc, 0, 1)
    
    # 验证集Accuracy（略低于训练集）
    val_acc = train_acc * 0.95 + np.random.normal(0, 0.015, num_epochs)
    val_acc = np.clip(val_acc, 0, 1)
    
    return {
        "epochs": epochs.tolist(),
        "train_loss": train_loss.tolist(),
        "val_loss": val_loss.tolist(),
        "train_acc": train_acc.tolist(),
        "val_acc": val_acc.tolist(),
        "final_val_acc": float(val_acc[-1]),
        "best_val_acc": float(np.max(val_acc)),
        "convergence_epoch": int(np.argmax(val_acc > 0.95 * np.max(val_acc)))
    }


def compare_convergence_speed(curves_dict: Dict[str, Dict]) -> Dict:
    """
    对比不同模型的收敛速度
    
    Args:
        curves_dict: 模型名 -> 训练曲线的字典
    
    Returns:
        comparison: 收敛速度对比结果
    """
    comparison = {}
    
    for model_name, curves in curves_dict.items():
        val_acc = np.array(curves['val_acc'])
        best_acc = np.max(val_acc)
        
        # 达到90%最佳精度的epoch
        target_acc = 0.90 * best_acc
        epoch_90 = np.where(val_acc >= target_acc)[0]
        epoch_90 = int(epoch_90[0]) if len(epoch_90) > 0 else len(val_acc)
        
        # 达到95%最佳精度的epoch
        target_acc = 0.95 * best_acc
        epoch_95 = np.where(val_acc >= target_acc)[0]
        epoch_95 = int(epoch_95[0]) if len(epoch_95) > 0 else len(val_acc)
        
        comparison[model_name] = {
            "best_acc": float(best_acc),
            "epoch_to_90": epoch_90,
            "epoch_to_95": epoch_95,
            "final_acc": float(val_acc[-1])
        }
    
    return comparison


def get_data_efficiency_curve(model_type: str) -> Dict:
    """
    生成数据效率曲线
    
    展示不同数据量下的模型性能
    
    Args:
        model_type: "CNN" 或 "Transformer"
    
    Returns:
        curve: 数据效率曲线
    """
    data_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    if model_type == "CNN":
        # CNN在小数据上表现好
        base_acc = 0.65
        accuracies = [
            0.65, 0.72, 0.76, 0.82, 0.86, 0.88
        ]
    else:  # Transformer
        # ViT在小数据上表现差，大数据上表现好
        base_acc = 0.50
        accuracies = [
            0.50, 0.60, 0.68, 0.78, 0.85, 0.91
        ]
    
    # 添加一些随机波动
    accuracies = [acc + np.random.normal(0, 0.01) for acc in accuracies]
    
    return {
        "data_ratios": data_ratios,
        "accuracies": accuracies
    }


def get_comparison_recommendations(
    data_size: str,
    compute_budget: str,
    task_type: str
) -> Dict:
    """
    根据条件推荐模型
    
    Args:
        data_size: "small", "medium", "large"
        compute_budget: "low", "medium", "high"
        task_type: "classification", "detection", "segmentation"
    
    Returns:
        recommendations: 推荐结果
    """
    recommendations = {
        "primary": None,
        "alternative": None,
        "reason": ""
    }
    
    # 决策逻辑
    if data_size == "small":
        if compute_budget == "low":
            recommendations["primary"] = "MobileNet-V2"
            recommendations["alternative"] = "ResNet-18"
            recommendations["reason"] = "小数据集 + 低算力 → 轻量级CNN最佳"
        else:
            recommendations["primary"] = "ResNet-18"
            recommendations["alternative"] = "ResNet-50"
            recommendations["reason"] = "小数据集 → CNN的归纳偏置优势明显"
    
    elif data_size == "medium":
        if compute_budget == "low":
            recommendations["primary"] = "MobileNet-V2"
            recommendations["alternative"] = "ViT-Tiny"
            recommendations["reason"] = "中等数据 + 低算力 → 轻量级模型"
        else:
            recommendations["primary"] = "ResNet-50"
            recommendations["alternative"] = "ViT-Small"
            recommendations["reason"] = "中等数据集 → CNN和ViT都可以，CNN更稳定"
    
    else:  # large
        if compute_budget == "low":
            recommendations["primary"] = "ResNet-50"
            recommendations["alternative"] = "ViT-Tiny"
            recommendations["reason"] = "大数据 + 低算力 → 平衡选择"
        elif compute_budget == "medium":
            recommendations["primary"] = "ViT-Small"
            recommendations["alternative"] = "ResNet-50"
            recommendations["reason"] = "大数据 + 中等算力 → ViT开始显现优势"
        else:  # high
            recommendations["primary"] = "ViT-Base"
            recommendations["alternative"] = "ViT-Small"
            recommendations["reason"] = "大数据 + 高算力 → ViT表现最佳"
    
    return recommendations


if __name__ == "__main__":
    print("=" * 60)
    print("模型对比工具测试")
    print("=" * 60)
    
    # 测试模型信息获取
    print("\n### 模型信息测试 ###")
    for model_name in ["ResNet-18", "ViT-Tiny"]:
        info = get_model_info(model_name)
        print(f"\n{model_name}:")
        print(f"  类型: {info['type']}")
        print(f"  参数量: {info['params']}M")
        print(f"  FLOPs: {info['flops']} GFLOPs")
        print(f"  归纳偏置: {info['inductive_bias']}")
    
    # 测试训练曲线生成
    print("\n### 训练曲线生成测试 ###")
    for model_type in ["CNN", "Transformer"]:
        for dataset_size in ["small", "large"]:
            curves = generate_training_curve(model_type, dataset_size, num_epochs=50)
            print(f"\n{model_type} on {dataset_size} dataset:")
            print(f"  最终验证精度: {curves['final_val_acc']:.4f}")
            print(f"  最佳验证精度: {curves['best_val_acc']:.4f}")
            print(f"  收敛epoch: {curves['convergence_epoch']}")
    
    # 测试收敛速度对比
    print("\n### 收敛速度对比测试 ###")
    curves_dict = {
        "ResNet-18": generate_training_curve("CNN", "medium", num_epochs=100),
        "ViT-Tiny": generate_training_curve("Transformer", "medium", num_epochs=100)
    }
    comparison = compare_convergence_speed(curves_dict)
    for model_name, stats in comparison.items():
        print(f"\n{model_name}:")
        print(f"  最佳精度: {stats['best_acc']:.4f}")
        print(f"  达到90%最佳精度: {stats['epoch_to_90']} epochs")
        print(f"  达到95%最佳精度: {stats['epoch_to_95']} epochs")
    
    # 测试数据效率曲线
    print("\n### 数据效率曲线测试 ###")
    for model_type in ["CNN", "Transformer"]:
        curve = get_data_efficiency_curve(model_type)
        print(f"\n{model_type}:")
        print(f"  10%数据: {curve['accuracies'][0]:.4f}")
        print(f"  100%数据: {curve['accuracies'][-1]:.4f}")
        print(f"  提升: {(curve['accuracies'][-1] - curve['accuracies'][0]):.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
