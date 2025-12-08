"""
动态参数建议器
Dynamic Parameter Suggester

基于输入复杂度和用户配置提供智能的参数建议
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ParameterSuggester:
    """动态参数建议器

    基于输入复杂度、数据集特征等提供智能的参数建议
    """

    @staticmethod
    def suggest_gnn_layers(
        num_nodes: int, feature_dim: int, task_complexity: str = "medium"
    ) -> Dict[str, Any]:
        """
        建议GNN的层数和隐藏维度

        Args:
            num_nodes: 节点数量
            feature_dim: 特征维度
            task_complexity: 任务复杂度 ("simple", "medium", "complex")

        Returns:
            建议的参数字典
        """
        # 基于图的大小建议层数
        if num_nodes <= 50:
            base_layers = 2
        elif num_nodes <= 500:
            base_layers = 3
        else:
            base_layers = 4

        # 根据任务复杂度调整
        complexity_multipliers = {"simple": 0.8, "medium": 1.0, "complex": 1.3}

        suggested_layers = max(
            2, int(base_layers * complexity_multipliers[task_complexity])
        )

        # 建议隐藏维度（基于特征维度）
        if feature_dim <= 8:
            hidden_dims = [32, 64]
        elif feature_dim <= 64:
            hidden_dims = [64, 128]
        elif feature_dim <= 256:
            hidden_dims = [128, 256]
        else:
            hidden_dims = [256, 512]

        # 根据任务复杂度调整隐藏维度
        hidden_dims = [
            int(dim * complexity_multipliers[task_complexity]) for dim in hidden_dims
        ]

        return {
            "num_layers": suggested_layers,
            "hidden_dims": hidden_dims[:suggested_layers],
            "dropout": 0.2 if task_complexity == "simple" else 0.5,
            "learning_rate": 0.01 if task_complexity == "simple" else 0.001,
        }

    @staticmethod
    def suggest_rnn_params(
        sequence_length: int, input_size: int, task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        建议RNN/LSTM的参数

        Args:
            sequence_length: 序列长度
            input_size: 输入维度
            task_type: 任务类型 ("classification", "regression", "generation")

        Returns:
            建议的参数字典
        """
        # 基于序列长度建议隐藏维度
        if sequence_length <= 20:
            base_hidden = 64
        elif sequence_length <= 100:
            base_hidden = 128
        elif sequence_length <= 500:
            base_hidden = 256
        else:
            base_hidden = 512

        # 根据输入维度调整
        input_factor = min(2.0, max(0.5, input_size / 100))
        hidden_size = int(base_hidden * input_factor)

        # 建议层数
        if sequence_length <= 50:
            num_layers = 1 if task_type == "regression" else 2
        elif sequence_length <= 200:
            num_layers = 2
        else:
            num_layers = 3

        # 根据任务类型调整参数
        task_params = {
            "classification": {
                "dropout": 0.3,
                "learning_rate": 0.001,
                "bidirectional": True,
            },
            "regression": {
                "dropout": 0.2,
                "learning_rate": 0.01,
                "bidirectional": False,
            },
            "generation": {
                "dropout": 0.4,
                "learning_rate": 0.0001,
                "bidirectional": False,
            },
        }

        params = task_params.get(task_type, task_params["classification"])
        params.update(
            {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "input_size": input_size,
            }
        )

        return params

    @staticmethod
    def suggest_resnet_params(
        input_size: int, num_classes: int, model_complexity: str = "medium"
    ) -> Dict[str, Any]:
        """
        建议ResNet的参数

        Args:
            input_size: 输入图像尺寸
            num_classes: 类别数
            model_complexity: 模型复杂度 ("light", "medium", "heavy")

        Returns:
            建议的参数字典
        """
        # 基于输入尺寸建议基础通道数
        if input_size <= 32:
            base_channels = 32
        elif input_size <= 64:
            base_channels = 64
        elif input_size <= 128:
            base_channels = 128
        else:
            base_channels = 256

        # 根据复杂度调整
        complexity_multipliers = {"light": 0.5, "medium": 1.0, "heavy": 2.0}

        base_channels = int(base_channels * complexity_multipliers[model_complexity])

        # 建议层数
        layer_configs = {
            "light": [2, 2, 2, 2],  # ResNet-18 like
            "medium": [3, 4, 6, 3],  # ResNet-34 like
            "heavy": [3, 4, 23, 3],  # ResNet-50 like
        }

        layers = layer_configs[model_complexity]

        return {
            "base_channels": base_channels,
            "layers": layers,
            "num_classes": num_classes,
            "input_size": input_size,
        }

    @staticmethod
    def suggest_vit_params(
        img_size: int, num_classes: int, model_size: str = "base"
    ) -> Dict[str, Any]:
        """
        建议Vision Transformer的参数

        Args:
            img_size: 图像尺寸
            num_classes: 类别数
            model_size: 模型大小 ("tiny", "small", "base", "large")

        Returns:
            建议的参数字典
        """
        # 标准ViT配置
        vit_configs = {
            "tiny": {
                "patch_size": 16,
                "embed_dim": 192,
                "num_heads": 3,
                "num_layers": 12,
                "mlp_ratio": 4,
            },
            "small": {
                "patch_size": 16,
                "embed_dim": 384,
                "num_heads": 6,
                "num_layers": 12,
                "mlp_ratio": 4,
            },
            "base": {
                "patch_size": 16,
                "embed_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "mlp_ratio": 4,
            },
            "large": {
                "patch_size": 16,
                "embed_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "mlp_ratio": 4,
            },
        }

        config = vit_configs[model_size].copy()

        # 根据图像大小调整patch size
        if img_size <= 64:
            config["patch_size"] = 8
        elif img_size >= 384:
            config["patch_size"] = 32

        config.update(
            {
                "img_size": img_size,
                "num_classes": num_classes,
                "num_patches": (img_size // config["patch_size"]) ** 2,
            }
        )

        return config

    @staticmethod
    def suggest_normalization_params(
        input_shape: Tuple[int, ...], batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        建议归一化层的参数

        Args:
            input_shape: 输入形状
            batch_size: 批次大小（可选）

        Returns:
            建议的参数字典
        """
        if len(input_shape) == 4:  # 图像数据 (B, C, H, W)
            C, H, W = input_shape[1], input_shape[2], input_shape[3]

            # 建议分组数（对于GroupNorm）
            if C <= 8:
                suggested_groups = max(2, C // 2)
            elif C <= 32:
                suggested_groups = 8
            elif C <= 64:
                suggested_groups = 16
            else:
                suggested_groups = 32

            # 建议使用的归一化类型
            if batch_size and batch_size < 8:
                # 小批次建议使用GroupNorm或LayerNorm
                recommended_norm = "GroupNorm"
            elif H * W <= 32 * 32:  # 小特征图
                recommended_norm = "BatchNorm"
            else:
                recommended_norm = "BatchNorm"

            return {
                "num_features": C,
                "recommended_norm": recommended_norm,
                "suggested_groups": suggested_groups,
                "eps": 1e-5,
                "momentum": 0.1,
            }

        elif len(input_shape) == 3:  # 序列数据 (B, T, D)
            T, D = input_shape[1], input_shape[2]

            return {"normalized_shape": D, "recommended_norm": "LayerNorm", "eps": 1e-5}

        else:
            return {"recommended_norm": "LayerNorm", "eps": 1e-5}

    @staticmethod
    def calculate_memory_estimate(
        model_config: Dict, batch_size: int = 1, input_dtype: str = "float32"
    ) -> Dict[str, float]:
        """
        估算模型内存使用

        Args:
            model_config: 模型配置
            batch_size: 批次大小
            input_dtype: 输入数据类型

        Returns:
            内存估算字典（MB）
        """
        # 数据类型字节数
        dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}

        bytes_per_element = dtype_bytes.get(input_dtype, 4)

        # 估算参数内存
        total_params = model_config.get("total_params", 0)
        param_memory = total_params * bytes_per_element / (1024**2)  # MB

        # 估算激活内存（粗略估算）
        if "input_shape" in model_config:
            input_elements = np.prod(model_config["input_shape"])
            activation_memory = (
                input_elements * batch_size * bytes_per_element / (1024**2)
            )
        else:
            activation_memory = 0

        # 梯度内存（与参数相同）
        gradient_memory = param_memory

        # 优化器状态（假设使用Adam，2倍参数）
        optimizer_memory = param_memory * 2

        total_memory = (
            param_memory + activation_memory + gradient_memory + optimizer_memory
        )

        return {
            "parameters": param_memory,
            "activations": activation_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "total": total_memory,
        }


def get_suggested_params(module_type: str, **kwargs) -> Dict[str, Any]:
    """
    获取参数建议

    Args:
        module_type: 模块类型 ("gnn", "rnn", "resnet", "vit", "normalization")
        **kwargs: 相关参数

    Returns:
        建议的参数字典
    """
    suggester = ParameterSuggester()

    if module_type == "gnn":
        return suggester.suggest_gnn_layers(
            kwargs.get("num_nodes", 8),
            kwargs.get("feature_dim", 3),
            kwargs.get("task_complexity", "medium"),
        )
    elif module_type == "rnn":
        return suggester.suggest_rnn_params(
            kwargs.get("sequence_length", 20),
            kwargs.get("input_size", 10),
            kwargs.get("task_type", "classification"),
        )
    elif module_type == "resnet":
        return suggester.suggest_resnet_params(
            kwargs.get("input_size", 224),
            kwargs.get("num_classes", 10),
            kwargs.get("model_complexity", "medium"),
        )
    elif module_type == "vit":
        return suggester.suggest_vit_params(
            kwargs.get("img_size", 224),
            kwargs.get("num_classes", 10),
            kwargs.get("model_size", "base"),
        )
    elif module_type == "normalization":
        return suggester.suggest_normalization_params(
            kwargs.get("input_shape", (1, 64, 32, 32)), kwargs.get("batch_size")
        )
    else:
        raise ValueError(f"不支持的模块类型: {module_type}")
