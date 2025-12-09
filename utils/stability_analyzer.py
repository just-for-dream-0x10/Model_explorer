"""
数值稳定性分析器
Numerical Stability Analyzer

检测神经网络训练时的数值稳定性问题
包括：梯度消失/爆炸、激活值异常、权重异常

新增功能（Phase 3）:
- 实时梯度检测
- 初始化方案推荐
- 峰值内存预测
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


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
        self.gradient_status = "未检测"  # 正常、消失、爆炸、包含NaN/Inf
        self.weight_status = "未检测"  # 正常、异常、未初始化

        # 问题描述和建议
        self.issues = []
        self.recommendations = []


def check_activation_stability(
    activations: torch.Tensor,
    threshold_large: float = 100.0,
    threshold_small: float = 1e-3,
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
        "recommendations": [],
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
        result["issues"].append(
            f"激活值范围过大: [{result['min']:.2f}, {result['max']:.2f}]"
        )
        result["recommendations"].append("添加BatchNorm或LayerNorm")
        result["recommendations"].append("使用ReLU代替Sigmoid/Tanh")
        result["recommendations"].append("检查权重初始化方案")

    elif abs(result["mean"]) < threshold_small and result["std"] < threshold_small:
        result["status"] = "异常小"
        result["issues"].append(
            f"激活值过小: mean={result['mean']:.2e}, std={result['std']:.2e}"
        )
        result["recommendations"].append("可能存在梯度消失")
        result["recommendations"].append("检查激活函数（避免Sigmoid）")
        result["recommendations"].append("使用残差连接")

    return result


def check_gradient_stability(
    gradients: torch.Tensor,
    threshold_vanish: float = 1e-7,
    threshold_explode: float = 10.0,
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
        "recommendations": [],
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
        result["recommendations"].append(
            "使用梯度裁剪: torch.nn.utils.clip_grad_norm_()"
        )
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
        "recommendations": [],
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
    model: nn.Module, input_data: torch.Tensor, num_steps: int = 10
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
            if hasattr(module, "weight") and module.weight is not None:
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
    gradient_vanish_layers = [
        info for info in layers_info if info.gradient_status == "梯度消失"
    ]
    gradient_explode_layers = [
        info for info in layers_info if info.gradient_status == "梯度爆炸"
    ]
    activation_issue_layers = [
        info for info in layers_info if info.activation_status not in ["正常", "未检测"]
    ]

    result = {
        "layers": layers_info,
        "summary": {
            "total_layers": len(layers_info),
            "total_issues": total_issues,
            "problem_layers": len(problem_layers),
            "gradient_vanish_count": len(gradient_vanish_layers),
            "gradient_explode_count": len(gradient_explode_layers),
            "activation_issue_count": len(activation_issue_layers),
        },
        "problem_layers": problem_layers,
        "gradient_vanish_layers": gradient_vanish_layers,
        "gradient_explode_layers": gradient_explode_layers,
        "activation_issue_layers": activation_issue_layers,
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
    print(
        f"正常激活值: 状态={result['status']}, mean={result['mean']:.4f}, std={result['std']:.4f}"
    )

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


# ==================== Phase 3 新增功能 ====================


def detect_gradient_flow_realtime(
    model: nn.Module, sample_input: torch.Tensor, loss_fn: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    实时检测梯度流动情况

    参数:
        model: 神经网络模型
        sample_input: 样本输入
        loss_fn: 损失函数（如果为None，使用输出的sum）

    返回:
        包含梯度统计和诊断信息的字典
    """
    model.train()

    # 清除之前的梯度
    model.zero_grad()

    # 前向传播
    output = model(sample_input)

    # 计算损失
    if loss_fn is not None:
        if output.dim() > 1 and output.size(-1) > 1:
            # 分类任务，创建假标签
            target = torch.zeros(output.size(0), dtype=torch.long)
            loss = loss_fn(output, target)
        else:
            loss = output.sum()
    else:
        loss = output.sum()

    # 反向传播
    loss.backward()

    # 收集梯度信息
    gradient_info = {}
    gradient_norms = {}
    layer_gradients = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            grad_max = grad.abs().max().item()

            gradient_norms[name] = grad_norm
            layer_gradients[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "max": grad_max,
                "shape": tuple(grad.shape),
                "has_nan": torch.isnan(grad).any().item(),
                "has_inf": torch.isinf(grad).any().item(),
            }

    # 检测梯度问题
    vanishing_threshold = 1e-7
    exploding_threshold = 100.0

    vanishing_layers = {
        k: v for k, v in gradient_norms.items() if v < vanishing_threshold and v > 0
    }

    exploding_layers = {
        k: v for k, v in gradient_norms.items() if v > exploding_threshold
    }

    nan_inf_layers = {
        k: v for k, v in layer_gradients.items() if v["has_nan"] or v["has_inf"]
    }

    # 计算统计信息
    if gradient_norms:
        grad_norms_list = list(gradient_norms.values())
        gradient_info["statistics"] = {
            "mean_norm": np.mean(grad_norms_list),
            "std_norm": np.std(grad_norms_list),
            "min_norm": np.min(grad_norms_list),
            "max_norm": np.max(grad_norms_list),
            "median_norm": np.median(grad_norms_list),
        }
    else:
        gradient_info["statistics"] = {}

    # 诊断结果
    gradient_info["all_gradients"] = layer_gradients
    gradient_info["gradient_norms"] = gradient_norms
    gradient_info["vanishing"] = vanishing_layers
    gradient_info["exploding"] = exploding_layers
    gradient_info["nan_inf"] = nan_inf_layers
    gradient_info["healthy"] = (
        len(vanishing_layers) == 0
        and len(exploding_layers) == 0
        and len(nan_inf_layers) == 0
    )

    # 生成建议
    recommendations = []

    if vanishing_layers:
        recommendations.append(
            {
                "issue": "梯度消失",
                "affected_layers": list(vanishing_layers.keys()),
                "severity": "high",
                "suggestions": [
                    "使用 ReLU 或 LeakyReLU 激活函数",
                    "使用残差连接（ResNet）",
                    "使用 BatchNorm 或 LayerNorm",
                    "减小网络深度",
                    "使用 Xavier/He 初始化",
                ],
            }
        )

    if exploding_layers:
        recommendations.append(
            {
                "issue": "梯度爆炸",
                "affected_layers": list(exploding_layers.keys()),
                "severity": "critical",
                "suggestions": [
                    "降低学习率",
                    "使用梯度裁剪 (gradient clipping)",
                    "使用 BatchNorm",
                    "检查权重初始化",
                    "使用更小的权重初始化标准差",
                ],
            }
        )

    if nan_inf_layers:
        recommendations.append(
            {
                "issue": "数值溢出 (NaN/Inf)",
                "affected_layers": list(nan_inf_layers.keys()),
                "severity": "critical",
                "suggestions": [
                    "显著降低学习率",
                    "使用梯度裁剪",
                    "检查数据预处理（归一化）",
                    "使用混合精度训练",
                    "检查损失函数实现",
                ],
            }
        )

    gradient_info["recommendations"] = recommendations

    return gradient_info


def recommend_initialization(
    layer: nn.Module, layer_name: str = "", activation: str = "relu"
) -> Dict[str, Any]:
    """
    推荐合适的初始化方案

    参数:
        layer: 神经网络层
        layer_name: 层名称
        activation: 激活函数类型

    返回:
        初始化推荐信息
    """
    layer_type = layer.__class__.__name__
    recommendation = {
        "layer_name": layer_name or layer_type,
        "layer_type": layer_type,
        "activation": activation,
    }

    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        # 根据激活函数推荐初始化
        if activation.lower() in ["relu", "leakyrelu", "elu"]:
            recommendation["method"] = "kaiming_normal"
            recommendation["reason"] = "ReLU系列激活函数的最佳实践"
            recommendation["code"] = (
                f"nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')"
            )
            recommendation["description"] = "He初始化，考虑了ReLU会将负值置零的特性"

        elif activation.lower() in ["sigmoid", "tanh"]:
            recommendation["method"] = "xavier_uniform"
            recommendation["reason"] = "Sigmoid/Tanh的最佳实践"
            recommendation["code"] = f"nn.init.xavier_uniform_(layer.weight)"
            recommendation["description"] = (
                "Xavier初始化，保持方差在前向和反向传播中一致"
            )

        elif activation.lower() in ["gelu", "silu", "swish"]:
            recommendation["method"] = "xavier_normal"
            recommendation["reason"] = "平滑激活函数的推荐方案"
            recommendation["code"] = f"nn.init.xavier_normal_(layer.weight)"
            recommendation["description"] = "Xavier初始化的正态分布版本"

        else:
            recommendation["method"] = "default"
            recommendation["reason"] = "使用PyTorch默认初始化"
            recommendation["code"] = "# 使用默认初始化"
            recommendation["description"] = "PyTorch的默认uniform初始化"

        # 偏置初始化
        if hasattr(layer, "bias") and layer.bias is not None:
            recommendation["bias_init"] = {
                "method": "zeros",
                "code": "nn.init.zeros_(layer.bias)",
                "reason": "偏置通常初始化为0",
            }

    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        recommendation["method"] = "ones_and_zeros"
        recommendation["reason"] = "BatchNorm的标准初始化"
        recommendation["code"] = (
            "nn.init.ones_(layer.weight)\n" "nn.init.zeros_(layer.bias)"
        )
        recommendation["description"] = "weight(gamma)初始化为1，bias(beta)初始化为0"

    elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
        recommendation["method"] = "orthogonal"
        recommendation["reason"] = "RNN的最佳实践"
        recommendation["code"] = (
            "for name, param in layer.named_parameters():\n"
            "    if 'weight_ih' in name:\n"
            "        nn.init.xavier_uniform_(param)\n"
            "    elif 'weight_hh' in name:\n"
            "        nn.init.orthogonal_(param)"
        )
        recommendation["description"] = "输入权重用Xavier，隐藏权重用正交初始化"

    else:
        recommendation["method"] = "not_applicable"
        recommendation["reason"] = "该层类型通常不需要特殊初始化"
        recommendation["code"] = "# 不需要特殊初始化"
        recommendation["description"] = f"{layer_type}层通常使用默认初始化即可"

    return recommendation


def predict_peak_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    optimizer_type: str = "adam",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    预测训练时的峰值内存使用

    参数:
        model: 神经网络模型
        input_shape: 输入形状（不包含batch维度）
        batch_size: 批大小
        optimizer_type: 优化器类型 ('sgd', 'adam', 'adamw')
        dtype: 数据类型

    返回:
        内存预测信息
    """
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8,
        torch.int32: 4,
        torch.int64: 8,
    }.get(dtype, 4)

    # 计算参数内存
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * bytes_per_element / (1024**2)  # MB

    # 计算梯度内存（与参数相同）
    gradient_memory = param_memory

    # 计算优化器状态内存
    if optimizer_type.lower() in ["adam", "adamw"]:
        # Adam 需要两个状态：momentum 和 variance（每个与参数大小相同）
        optimizer_memory = param_memory * 2
    elif optimizer_type.lower() == "sgd":
        # SGD with momentum 需要一个状态
        optimizer_memory = param_memory
    else:
        optimizer_memory = 0

    # 估算前向传播激活值内存
    # 简化估算：假设每层的激活值大小逐渐减小
    try:
        # 创建样本输入
        full_input_shape = (batch_size,) + input_shape
        sample_input = torch.randn(full_input_shape, dtype=dtype)

        # 统计激活值
        activation_memory = 0
        hooks = []

        def hook_fn(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * bytes_per_element / (1024**2)

        # 注册hooks
        for module in model.modules():
            if len(list(module.children())) == 0:  # 只处理叶子模块
                hooks.append(module.register_forward_hook(hook_fn))

        # 前向传播
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)

        # 清理hooks
        for hook in hooks:
            hook.remove()

    except Exception as e:
        # 如果出错，使用经验公式
        input_elements = batch_size * np.prod(input_shape)
        activation_memory = (
            input_elements * bytes_per_element * 10 / (1024**2)
        )  # 粗略估计

    # 计算反向传播内存（通常是前向的2-3倍）
    backward_memory = activation_memory * 2.5

    # 峰值内存 = 参数 + 梯度 + 优化器状态 + 前向激活 + 反向激活
    peak_memory = (
        param_memory
        + gradient_memory
        + optimizer_memory
        + activation_memory
        + backward_memory
    )

    memory_info = {
        "total_peak": peak_memory,
        "breakdown": {
            "parameters": param_memory,
            "gradients": gradient_memory,
            "optimizer_states": optimizer_memory,
            "forward_activations": activation_memory,
            "backward_activations": backward_memory,
        },
        "parameter_count": param_count,
        "batch_size": batch_size,
        "optimizer_type": optimizer_type,
        "dtype": str(dtype),
        "bytes_per_element": bytes_per_element,
    }

    # 生成建议
    recommendations = []

    if peak_memory > 1000:  # > 1GB
        recommendations.append(
            {
                "issue": "内存占用较大",
                "severity": "medium",
                "suggestions": [
                    f"减小批大小（当前: {batch_size}）",
                    "使用梯度累积",
                    "使用混合精度训练（AMP）",
                    "使用梯度检查点（gradient checkpointing）",
                ],
            }
        )

    if peak_memory > 4000:  # > 4GB
        recommendations.append(
            {
                "issue": "内存占用很大",
                "severity": "high",
                "suggestions": [
                    "强烈建议减小批大小",
                    "使用混合精度训练（可节省50%内存）",
                    "考虑使用模型并行",
                    "使用梯度检查点",
                ],
            }
        )

    if optimizer_type.lower() in ["adam", "adamw"]:
        recommendations.append(
            {
                "issue": "Adam优化器内存开销大",
                "severity": "info",
                "suggestions": [
                    f"Adam需要2倍参数内存存储状态（{optimizer_memory:.1f} MB）",
                    "可以考虑使用SGD（内存减半）",
                    "或使用Adafactor等内存优化的优化器",
                ],
            }
        )

    memory_info["recommendations"] = recommendations

    # 生成不同配置下的内存对比（简化版，避免递归）
    memory_info["memory_comparison"] = {
        "current": peak_memory,
        "half_batch": peak_memory * 0.5 if batch_size > 1 else peak_memory,
        "mixed_precision": peak_memory * 0.5,  # 混合精度约节省50%
        "sgd_optimizer": peak_memory
        - optimizer_memory
        + param_memory,  # SGD只需1倍参数内存
    }

    return memory_info


def analyze_numerical_stability(
    model: nn.Module, sample_input: torch.Tensor
) -> Dict[str, Any]:
    """
    综合分析数值稳定性

    结合梯度检测、初始化推荐和内存预测

    参数:
        model: 神经网络模型
        sample_input: 样本输入

    返回:
        综合分析结果
    """
    analysis = {}

    # 1. 梯度流动检测
    try:
        gradient_info = detect_gradient_flow_realtime(model, sample_input)
        analysis["gradient_flow"] = gradient_info
    except Exception as e:
        analysis["gradient_flow"] = {"error": str(e)}

    # 2. 初始化推荐
    initialization_recommendations = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LSTM)):
            rec = recommend_initialization(module, name, activation="relu")
            initialization_recommendations.append(rec)

    analysis["initialization"] = initialization_recommendations

    # 3. 内存预测
    try:
        input_shape = tuple(sample_input.shape[1:])  # 去掉batch维度
        batch_size = sample_input.shape[0]
        memory_info = predict_peak_memory(model, input_shape, batch_size)
        analysis["memory"] = memory_info
    except Exception as e:
        analysis["memory"] = {"error": str(e)}

    # 4. 整体健康评分
    health_score = 100
    issues = []

    if "gradient_flow" in analysis and not analysis["gradient_flow"].get(
        "healthy", True
    ):
        health_score -= 30
        issues.append("梯度流动异常")

    if "memory" in analysis:
        peak_mem = analysis["memory"].get("total_peak", 0)
        if peak_mem > 4000:
            health_score -= 20
            issues.append("内存占用过大")
        elif peak_mem > 1000:
            health_score -= 10
            issues.append("内存占用较大")

    analysis["overall"] = {
        "health_score": max(0, health_score),
        "status": (
            "healthy"
            if health_score >= 80
            else ("warning" if health_score >= 60 else "critical")
        ),
        "issues": issues,
    }

    return analysis


# ==================== 辅助函数 ====================


def format_memory_size(size_mb: float) -> str:
    """格式化内存大小显示"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.2f} GB"


def get_gradient_health_emoji(gradient_info: Dict[str, Any]) -> str:
    """获取梯度健康状态的emoji"""
    if gradient_info.get("healthy", False):
        return "✅"
    elif gradient_info.get("nan_inf"):
        return "🔴"
    elif gradient_info.get("exploding"):
        return "🟠"
    elif gradient_info.get("vanishing"):
        return "🟡"
    else:
        return "⚪"
