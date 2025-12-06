"""
数值稳定性分析器
Numerical Stability Analyzer

检测神经网络训练时的数值稳定性问题
包括：梯度消失/爆炸、激活值异常、权重异常
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class LayerStabilityInfo:
    """单层稳定性信息"""
    def __init__(self, name: str, layer_type: str):
        self.name = name
        self.layer_type = layer_type
        
        # 激活值统计
        self.activation_mean = 0.0
        self.activation_std = 0.0
        self.activation_min = 0.0
        self.activation_max = 0.0
        self.activation_range = 0.0
        
        # 梯度统计
        self.gradient_mean = 0.0
        self.gradient_std = 0.0
        self.gradient_norm = 0.0
        self.gradient_max = 0.0
        
        # 权重统计
        self.weight_mean = 0.0
        self.weight_std = 0.0
        self.weight_norm = 0.0
        
        # 稳定性状态
        self.activation_status = "未检测"  # 正常、异常大、异常小、包含NaN/Inf
        self.gradient_status = "未检测"     # 正常、消失、爆炸、包含NaN/Inf
        self.weight_status = "未检测"       # 正常、异常、未初始化
        
        # 问题描述和建议
        self.issues = []
        self.recommendations = []


def check_activation_stability(
    activations: torch.Tensor,
    threshold_large: float = 100.0,
    threshold_small: float = 1e-3
) -> Dict:
    """
    检查激活值的稳定性
    
    Args:
        activations: 激活值张量
        threshold_large: 异常大的阈值
        threshold_small: 异常小的阈值
    
    Returns:
        result: 稳定性检查结果
    """
    result = {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "range": 0.0,
        "has_nan": False,
        "has_inf": False,
        "status": "正常",
        "issues": [],
        "recommendations": []
    }
    
    # 检查NaN和Inf
    if torch.isnan(activations).any():
        result["has_nan"] = True
        result["status"] = "包含NaN"
        result["issues"].append("激活值包含NaN（Not a Number）")
        result["recommendations"].append("检查输入数据是否有NaN")
        result["recommendations"].append("检查权重初始化")
        result["recommendations"].append("降低学习率")
        return result
    
    if torch.isinf(activations).any():
        result["has_inf"] = True
        result["status"] = "包含Inf"
        result["issues"].append("激活值包含Inf（无穷大）")
        result["recommendations"].append("梯度爆炸导致数值溢出")
        result["recommendations"].append("使用梯度裁剪")
        result["recommendations"].append("降低学习率")
        return result
    
    # 统计信息
    result["mean"] = activations.mean().item()
    result["std"] = activations.std().item()
    result["min"] = activations.min().item()
    result["max"] = activations.max().item()
    result["range"] = result["max"] - result["min"]
    
    # 检查异常
    if abs(result["max"]) > threshold_large or abs(result["min"]) > threshold_large:
        result["status"] = "异常大"
        result["issues"].append(f"激活值范围过大: [{result['min']:.2f}, {result['max']:.2f}]")
        result["recommendations"].append("添加BatchNorm或LayerNorm")
        result["recommendations"].append("使用ReLU代替Sigmoid/Tanh")
        result["recommendations"].append("检查权重初始化方案")
    
    elif abs(result["mean"]) < threshold_small and result["std"] < threshold_small:
        result["status"] = "异常小"
        result["issues"].append(f"激活值过小: mean={result['mean']:.2e}, std={result['std']:.2e}")
        result["recommendations"].append("可能存在梯度消失")
        result["recommendations"].append("检查激活函数（避免Sigmoid）")
        result["recommendations"].append("使用残差连接")
    
    return result


def check_gradient_stability(
    gradients: torch.Tensor,
    threshold_vanish: float = 1e-7,
    threshold_explode: float = 10.0
) -> Dict:
    """
    检查梯度的稳定性
    
    Args:
        gradients: 梯度张量
        threshold_vanish: 梯度消失阈值
        threshold_explode: 梯度爆炸阈值
    
    Returns:
        result: 稳定性检查结果
    """
    result = {
        "mean": 0.0,
        "std": 0.0,
        "norm": 0.0,
        "max": 0.0,
        "has_nan": False,
        "has_inf": False,
        "status": "正常",
        "issues": [],
        "recommendations": []
    }
    
    # 检查NaN和Inf
    if torch.isnan(gradients).any():
        result["has_nan"] = True
        result["status"] = "包含NaN"
        result["issues"].append("梯度包含NaN")
        result["recommendations"].append("学习率可能过大")
        result["recommendations"].append("检查损失函数")
        result["recommendations"].append("使用梯度裁剪")
        return result
    
    if torch.isinf(gradients).any():
        result["has_inf"] = True
        result["status"] = "包含Inf"
        result["issues"].append("梯度包含Inf")
        result["recommendations"].append("梯度爆炸")
        result["recommendations"].append("使用梯度裁剪（clip_grad_norm）")
        result["recommendations"].append("降低学习率")
        return result
    
    # 统计信息
    result["mean"] = gradients.mean().item()
    result["std"] = gradients.std().item()
    result["norm"] = gradients.norm().item()
    result["max"] = gradients.abs().max().item()
    
    # 检查梯度消失
    if result["norm"] < threshold_vanish:
        result["status"] = "梯度消失"
        result["issues"].append(f"梯度范数过小: {result['norm']:.2e}")
        result["recommendations"].append("使用残差连接（ResNet）")
        result["recommendations"].append("使用ReLU/GELU激活函数")
        result["recommendations"].append("检查权重初始化（使用Xavier/He初始化）")
        result["recommendations"].append("添加BatchNorm")
    
    # 检查梯度爆炸
    elif result["norm"] > threshold_explode:
        result["status"] = "梯度爆炸"
        result["issues"].append(f"梯度范数过大: {result['norm']:.2f}")
        result["recommendations"].append("使用梯度裁剪: torch.nn.utils.clip_grad_norm_()")
        result["recommendations"].append("降低学习率（当前学习率×0.1）")
        result["recommendations"].append("使用BatchNorm稳定训练")
    
    return result


def check_weight_stability(weights: torch.Tensor) -> Dict:
    """
    检查权重的稳定性
    
    Args:
        weights: 权重张量
    
    Returns:
        result: 稳定性检查结果
    """
    result = {
        "mean": 0.0,
        "std": 0.0,
        "norm": 0.0,
        "has_nan": False,
        "has_inf": False,
        "status": "正常",
        "issues": [],
        "recommendations": []
    }
    
    # 检查NaN和Inf
    if torch.isnan(weights).any():
        result["has_nan"] = True
        result["status"] = "包含NaN"
        result["issues"].append("权重包含NaN")
        result["recommendations"].append("重新初始化模型")
        result["recommendations"].append("检查训练过程是否稳定")
        return result
    
    if torch.isinf(weights).any():
        result["has_inf"] = True
        result["status"] = "包含Inf"
        result["issues"].append("权重包含Inf")
        result["recommendations"].append("训练不稳定导致权重溢出")
        result["recommendations"].append("降低学习率")
        result["recommendations"].append("使用权重衰减（weight decay）")
        return result
    
    # 统计信息
    result["mean"] = weights.mean().item()
    result["std"] = weights.std().item()
    result["norm"] = weights.norm().item()
    
    # 检查权重是否合理
    if result["std"] < 1e-6:
        result["status"] = "未初始化或异常"
        result["issues"].append(f"权重标准差过小: {result['std']:.2e}")
        result["recommendations"].append("检查权重是否正确初始化")
        result["recommendations"].append("使用Xavier或He初始化")
    
    elif result["std"] > 10.0:
        result["status"] = "异常大"
        result["issues"].append(f"权重标准差过大: {result['std']:.2f}")
        result["recommendations"].append("权重可能增长失控")
        result["recommendations"].append("添加权重衰减（L2正则化）")
        result["recommendations"].append("降低学习率")
    
    return result


def analyze_model_stability(
    model: nn.Module,
    input_data: torch.Tensor,
    num_steps: int = 10
) -> Dict:
    """
    分析整个模型的数值稳定性
    
    Args:
        model: PyTorch模型
        input_data: 输入数据
        num_steps: 模拟训练步数
    
    Returns:
        result: 稳定性分析结果
    """
    model.train()
    
    layers_info = []
    
    # 注册hook收集激活值和梯度
    activations = {}
    gradients = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook
    
    def get_gradient(name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output[0], torch.Tensor):
                gradients[name] = grad_output[0].detach()
        return hook
    
    # 注册hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(get_activation(name)))
            hooks.append(module.register_backward_hook(get_gradient(name)))
    
    # 模拟多步训练
    for step in range(num_steps):
        model.zero_grad()
        
        # 前向传播
        output = model(input_data)
        
        # 构造损失
        target = torch.randn_like(output)
        loss = ((output - target) ** 2).mean()
        
        # 反向传播
        loss.backward()
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 分析每一层
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            info = LayerStabilityInfo(name, type(module).__name__)
            
            # 检查激活值
            if name in activations:
                act_result = check_activation_stability(activations[name])
                info.activation_mean = act_result["mean"]
                info.activation_std = act_result["std"]
                info.activation_min = act_result["min"]
                info.activation_max = act_result["max"]
                info.activation_range = act_result["range"]
                info.activation_status = act_result["status"]
                info.issues.extend(act_result["issues"])
                info.recommendations.extend(act_result["recommendations"])
            
            # 检查梯度
            if name in gradients:
                grad_result = check_gradient_stability(gradients[name])
                info.gradient_mean = grad_result["mean"]
                info.gradient_std = grad_result["std"]
                info.gradient_norm = grad_result["norm"]
                info.gradient_max = grad_result["max"]
                info.gradient_status = grad_result["status"]
                info.issues.extend(grad_result["issues"])
                info.recommendations.extend(grad_result["recommendations"])
            
            # 检查权重
            if hasattr(module, 'weight') and module.weight is not None:
                weight_result = check_weight_stability(module.weight)
                info.weight_mean = weight_result["mean"]
                info.weight_std = weight_result["std"]
                info.weight_norm = weight_result["norm"]
                info.weight_status = weight_result["status"]
                info.issues.extend(weight_result["issues"])
                info.recommendations.extend(weight_result["recommendations"])
            
            layers_info.append(info)
    
    # 汇总问题
    total_issues = sum(len(info.issues) for info in layers_info)
    problem_layers = [info for info in layers_info if len(info.issues) > 0]
    
    # 分类问题
    gradient_vanish_layers = [info for info in layers_info if info.gradient_status == "梯度消失"]
    gradient_explode_layers = [info for info in layers_info if info.gradient_status == "梯度爆炸"]
    activation_issue_layers = [info for info in layers_info if info.activation_status not in ["正常", "未检测"]]
    
    result = {
        "layers": layers_info,
        "summary": {
            "total_layers": len(layers_info),
            "total_issues": total_issues,
            "problem_layers": len(problem_layers),
            "gradient_vanish_count": len(gradient_vanish_layers),
            "gradient_explode_count": len(gradient_explode_layers),
            "activation_issue_count": len(activation_issue_layers)
        },
        "problem_layers": problem_layers,
        "gradient_vanish_layers": gradient_vanish_layers,
        "gradient_explode_layers": gradient_explode_layers,
        "activation_issue_layers": activation_issue_layers
    }
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("数值稳定性分析器测试")
    print("=" * 60)
    
    # 测试激活值检查
    print("\n### 激活值稳定性测试 ###")
    
    # 正常激活值
    normal_act = torch.randn(100) * 0.5
    result = check_activation_stability(normal_act)
    print(f"正常激活值: 状态={result['status']}, mean={result['mean']:.4f}, std={result['std']:.4f}")
    
    # 异常大的激活值
    large_act = torch.randn(100) * 100
    result = check_activation_stability(large_act)
    print(f"异常大激活值: 状态={result['status']}, 问题数={len(result['issues'])}")
    
    # 测试梯度检查
    print("\n### 梯度稳定性测试 ###")
    
    # 正常梯度
    normal_grad = torch.randn(100) * 0.1
    result = check_gradient_stability(normal_grad)
    print(f"正常梯度: 状态={result['status']}, norm={result['norm']:.4f}")
    
    # 梯度消失
    vanish_grad = torch.randn(100) * 1e-8
    result = check_gradient_stability(vanish_grad)
    print(f"梯度消失: 状态={result['status']}, norm={result['norm']:.2e}")
    
    # 梯度爆炸
    explode_grad = torch.randn(100) * 100
    result = check_gradient_stability(explode_grad)
    print(f"梯度爆炸: 状态={result['status']}, norm={result['norm']:.2f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
