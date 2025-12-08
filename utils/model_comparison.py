"""
模型对比工具模块
Model Comparison Utilities

提供模型对比、训练曲线生成等功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from .performance_predictor import (
    PerformancePredictor,
    create_model_config,
    create_dataset_config,
    create_training_config,
)


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
            "flops": 1.8,  # GFLOPs
            "depth": 18,
            "architecture": "残差网络",
            "inductive_bias": "强（平移不变性、局部性）",
            "best_for": "小到中等数据集、需要快速训练",
            "pretrain_dataset": "ImageNet-1K",
            "year": 2015,
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
            "year": 2015,
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
            "year": 2018,
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
            "year": 2020,
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
            "year": 2020,
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
            "year": 2020,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"未知模型: {model_name}")

    return model_configs[model_name]


def generate_training_curves(
    model_type: str = "CNN",
    dataset_size: str = "small",
    num_epochs: int = 100,
    seed: int = 42,
    model_params: int = None,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_classes: int = 10,
) -> Dict:
    """
    生成基于模型配置的动态训练曲线

    使用性能预测器基于实际模型配置生成合理的训练曲线
    替代硬编码的训练数据

    Args:
        model_type: "CNN" 或 "Transformer" 或 "RNN"
        dataset_size: "small" (10K), "medium" (50K), "large" (500K)
        num_epochs: 训练轮数
        seed: 随机种子
        model_params: 模型参数数量（如未提供则使用默认值）
        learning_rate: 学习率
        batch_size: 批次大小
        num_classes: 类别数

    Returns:
        curves: 包含训练曲线的字典
    """
    np.random.seed(seed)

    # 数据集大小映射
    dataset_size_map = {"small": 10000, "medium": 50000, "large": 500000}

    # 模型参数默认值
    model_params_map = {
        "CNN": 5e6,  # 5M参数
        "Transformer": 20e6,  # 20M参数
        "RNN": 2e6,  # 2M参数
    }

    # 模型深度默认值
    model_depth_map = {"CNN": 10, "Transformer": 12, "RNN": 3}

    # 使用提供的参数或默认值
    actual_dataset_size = dataset_size_map.get(dataset_size, 50000)
    actual_model_params = model_params or model_params_map.get(model_type, 5e6)
    actual_model_depth = model_depth_map.get(model_type, 10)

    # 创建配置
    model_config = create_model_config(
        model_type=model_type,
        num_params=actual_model_params,
        model_depth=actual_model_depth,
    )

    dataset_config = create_dataset_config(
        dataset_size=actual_dataset_size, num_classes=num_classes, data_complexity=0.5
    )

    training_config = create_training_config(
        learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs
    )

    # 使用性能预测器生成曲线
    predictor = PerformancePredictor()
    curves = predictor.predict_training_performance(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
    )

    return curves


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
        val_acc = np.array(curves["val_acc"])
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
            "final_acc": float(val_acc[-1]),
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
        accuracies = [0.65, 0.72, 0.76, 0.82, 0.86, 0.88]
    else:  # Transformer
        # ViT在小数据上表现差，大数据上表现好
        base_acc = 0.50
        accuracies = [0.50, 0.60, 0.68, 0.78, 0.85, 0.91]

    # 添加一些随机波动
    accuracies = [acc + np.random.normal(0, 0.01) for acc in accuracies]

    return {"data_ratios": data_ratios, "accuracies": accuracies}


def get_comparison_recommendations(
    data_size: str, compute_budget: str, task_type: str
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
    recommendations = {"primary": None, "alternative": None, "reason": ""}

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
        "ViT-Tiny": generate_training_curve("Transformer", "medium", num_epochs=100),
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
