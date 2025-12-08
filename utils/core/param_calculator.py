"""参数量计算器核心模块

提供各种网络层参数量的精确计算功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from ..exceptions import InvalidLayerConfigError, ComputationError


class ParameterCalculator:
    """参数量计算器

    提供各种网络层参数量和FLOPs的精确计算。
    """

    @staticmethod
    def calculate_conv2d_params(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        input_shape: Tuple[int, int, int] = None,
        use_bias: bool = True,
    ) -> Dict[str, Any]:
        """计算卷积层参数量和FLOPs

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            input_shape: 输入形状 (C, H, W)
            use_bias: 是否使用偏置

        Returns:
            包含参数量、FLOPs等信息的字典

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        # 验证输入参数
        ParameterCalculator._validate_positive_int(in_channels, "in_channels")
        ParameterCalculator._validate_positive_int(out_channels, "out_channels")
        ParameterCalculator._validate_positive_int(kernel_size, "kernel_size")
        ParameterCalculator._validate_positive_int(stride, "stride")
        ParameterCalculator._validate_non_negative_int(padding, "padding")

        if input_shape:
            if len(input_shape) != 3:
                raise InvalidLayerConfigError(
                    layer_type="conv2d",
                    param_name="input_shape",
                    param_value=input_shape,
                    expected_type="长度为3的元组",
                )

            C_in, H_in, W_in = input_shape
            if C_in != in_channels:
                raise InvalidLayerConfigError(
                    layer_type="conv2d",
                    param_name="input_shape[0]",
                    param_value=C_in,
                    expected_type=f"匹配in_channels={in_channels}",
                )

            # 计算输出尺寸
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            # 计算FLOPs
            flops_per_output = kernel_size * kernel_size * in_channels
            total_macs = flops_per_output * out_channels * H_out * W_out
            total_flops = 2 * total_macs

            if use_bias:
                total_flops += out_channels * H_out * W_out
        else:
            H_out = W_out = None
            total_flops = None

        # 计算参数量
        weight_params = out_channels * in_channels * kernel_size * kernel_size
        bias_params = out_channels if use_bias else 0
        total_params = weight_params + bias_params

        # 计算内存占用 (FP32)
        param_memory_mb = total_params * 4 / (1024**2)

        result = {
            "layer_type": "Conv2d",
            "parameters": {
                "weight": weight_params,
                "bias": bias_params,
                "total": total_params,
            },
            "param_memory_mb": param_memory_mb,
            "config": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "use_bias": use_bias,
            },
        }

        if input_shape:
            result.update(
                {
                    "input_shape": input_shape,
                    "output_shape": (out_channels, H_out, W_out),
                    "flops": {
                        "total": total_flops,
                        "macs": total_macs,
                        "readable": ParameterCalculator._format_flops(total_flops),
                    },
                    "output_memory_mb": out_channels * H_out * W_out * 4 / (1024**2),
                }
            )

        return result

    @staticmethod
    def calculate_linear_params(
        in_features: int, out_features: int, use_bias: bool = True
    ) -> Dict[str, Any]:
        """计算全连接层参数量

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            use_bias: 是否使用偏置

        Returns:
            包含参数量等信息的字典

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        # 验证输入参数
        ParameterCalculator._validate_positive_int(in_features, "in_features")
        ParameterCalculator._validate_positive_int(out_features, "out_features")

        # 计算参数量
        weight_params = in_features * out_features
        bias_params = out_features if use_bias else 0
        total_params = weight_params + bias_params

        # 计算FLOPs
        total_flops = 2 * weight_params

        # 计算内存占用
        param_memory_mb = total_params * 4 / (1024**2)

        return {
            "layer_type": "Linear",
            "parameters": {
                "weight": weight_params,
                "bias": bias_params,
                "total": total_params,
            },
            "param_memory_mb": param_memory_mb,
            "flops": {
                "total": total_flops,
                "readable": ParameterCalculator._format_flops(total_flops),
            },
            "config": {
                "in_features": in_features,
                "out_features": out_features,
                "use_bias": use_bias,
            },
        }

    @staticmethod
    def calculate_attention_params(
        embed_dim: int, num_heads: int, sequence_length: int = None
    ) -> Dict[str, Any]:
        """计算多头注意力层参数量

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            sequence_length: 序列长度（用于计算FLOPs）

        Returns:
            包含参数量等信息的字典

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        # 验证输入参数
        ParameterCalculator._validate_positive_int(embed_dim, "embed_dim")
        ParameterCalculator._validate_positive_int(num_heads, "num_heads")

        if embed_dim % num_heads != 0:
            raise InvalidLayerConfigError(
                layer_type="MultiHeadAttention",
                param_name="embed_dim",
                param_value=embed_dim,
                expected_type=f"能被num_heads={num_heads}整除",
            )

        head_dim = embed_dim // num_heads

        # QKV投影参数量
        qkv_params = embed_dim * embed_dim * 3  # Q, K, V各一个投影

        # 输出投影参数量
        output_params = embed_dim * embed_dim

        # 总参数量
        total_params = qkv_params + output_params

        # 计算FLOPs（如果提供了序列长度）
        if sequence_length:
            # QKV投影
            qkv_flops = 2 * qkv_params * sequence_length

            # 注意力计算 (缩放点积注意力)
            attn_flops = 2 * num_heads * sequence_length * sequence_length * head_dim

            # 输出投影
            output_flops = 2 * output_params * sequence_length

            total_flops = qkv_flops + attn_flops + output_flops
        else:
            total_flops = None

        # 计算内存占用
        param_memory_mb = total_params * 4 / (1024**2)

        result = {
            "layer_type": "MultiHeadAttention",
            "parameters": {
                "qkv_projection": qkv_params,
                "output_projection": output_params,
                "total": total_params,
            },
            "param_memory_mb": param_memory_mb,
            "config": {
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
        }

        if sequence_length:
            result.update(
                {
                    "sequence_length": sequence_length,
                    "flops": {
                        "total": total_flops,
                        "qkv_projection": qkv_flops,
                        "attention": attn_flops,
                        "output_projection": output_flops,
                        "readable": ParameterCalculator._format_flops(total_flops),
                    },
                }
            )

        return result

    @staticmethod
    def calculate_batchnorm_params(
        num_features: int, num_dims: int = 2
    ) -> Dict[str, Any]:
        """计算批归一化层参数量

        Args:
            num_features: 特征数
            num_dims: 维度数（1D, 2D, 3D）

        Returns:
            包含参数量等信息的字典

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        # 验证输入参数
        ParameterCalculator._validate_positive_int(num_features, "num_features")
        ParameterCalculator._validate_int_in_range(num_dims, 1, 3, "num_dims")

        # 批归一化参数：weight + bias + running_mean + running_var
        params_per_feature = 4
        total_params = num_features * params_per_feature

        # 计算内存占用
        param_memory_mb = total_params * 4 / (1024**2)

        return {
            "layer_type": f"BatchNorm{num_dims}d",
            "parameters": {"total": total_params, "per_feature": params_per_feature},
            "param_memory_mb": param_memory_mb,
            "config": {"num_features": num_features, "num_dims": num_dims},
        }

    @staticmethod
    def _validate_positive_int(value: int, param_name: str) -> None:
        """验证正整数参数

        Args:
            value: 参数值
            param_name: 参数名

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        if not isinstance(value, int) or value <= 0:
            raise InvalidLayerConfigError(
                layer_type="parameter_validation",
                param_name=param_name,
                param_value=value,
                expected_type="正整数",
            )

    @staticmethod
    def _validate_non_negative_int(value: int, param_name: str) -> None:
        """验证非负整数参数

        Args:
            value: 参数值
            param_name: 参数名

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        if not isinstance(value, int) or value < 0:
            raise InvalidLayerConfigError(
                layer_type="parameter_validation",
                param_name=param_name,
                param_value=value,
                expected_type="非负整数",
            )

    @staticmethod
    def _validate_int_in_range(
        value: int, min_val: int, max_val: int, param_name: str
    ) -> None:
        """验证范围内的整数参数

        Args:
            value: 参数值
            min_val: 最小值
            max_val: 最大值
            param_name: 参数名

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        if not isinstance(value, int) or not (min_val <= value <= max_val):
            raise InvalidLayerConfigError(
                layer_type="parameter_validation",
                param_name=param_name,
                param_value=value,
                expected_type=f"范围[{min_val}, {max_val}]内的整数",
            )

    @staticmethod
    def _format_flops(flops: int) -> str:
        """格式化FLOPs显示

        Args:
            flops: FLOPs数值

        Returns:
            格式化后的字符串
        """
        if flops >= 1e9:
            return f"{flops/1e9:.2f}G"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f}M"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f}K"
        else:
            return str(flops)
