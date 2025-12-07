"""网络层分析器

提供各种网络层的参数量、FLOPs和内存占用计算功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple


class LayerAnalyzer:
    """网络层分析器：计算参数量、FLOPs、内存占用"""

    @staticmethod
    def conv2d_analysis(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        input_shape: Tuple[int, int, int],
        use_bias: bool = True,
    ) -> Dict:
        """
        分析 Conv2d 层的计算细节

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            input_shape: (C, H, W)
            use_bias: 是否使用偏置

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        C_in, H_in, W_in = input_shape

        # 输出尺寸计算
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1

        # 参数量计算
        # 权重参数: out_channels × in_channels × kernel_size × kernel_size
        weight_params = out_channels * in_channels * kernel_size * kernel_size
        # 偏置参数: out_channels (如果使用)
        bias_params = out_channels if use_bias else 0
        total_params = weight_params + bias_params

        # FLOPs 计算
        # 每个输出位置需要: kernel_size² × in_channels 次乘法
        # 输出位置总数: out_channels × H_out × W_out
        macs_per_position = kernel_size * kernel_size * in_channels  # 乘加操作
        total_macs = macs_per_position * out_channels * H_out * W_out
        # 1 MAC = 2 FLOPs (1个乘法 + 1个加法)
        total_flops = 2 * total_macs

        # 如果有偏置，每个输出位置还需要1次加法
        if use_bias:
            total_flops += out_channels * H_out * W_out

        # 内存占用 (假设 FP32, 每个参数 4 bytes)
        param_memory_mb = (total_params * 4) / (1024**2)

        # 前向传播激活值内存
        input_memory = C_in * H_in * W_in * 4 / (1024**2)  # MB
        output_memory = out_channels * H_out * W_out * 4 / (1024**2)  # MB
        forward_memory_mb = input_memory + output_memory

        # 反向传播需要存储输入和输出的梯度，内存翻倍
        backward_memory_mb = forward_memory_mb * 2

        return {
            "layer_type": "Conv2d",
            "input_shape": (C_in, H_in, W_in),
            "output_shape": (out_channels, H_out, W_out),
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "parameters": {
                "weight": weight_params,
                "bias": bias_params,
                "total": total_params,
            },
            "flops": {
                "macs": total_macs,
                "total": total_flops,
                "macs_readable": (
                    f"{total_macs / 1e6:.2f}M"
                    if total_macs > 1e6
                    else f"{total_macs / 1e3:.2f}K"
                ),
                "flops_readable": (
                    f"{total_flops / 1e9:.2f}G"
                    if total_flops > 1e9
                    else f"{total_flops / 1e6:.2f}M"
                ),
            },
            "memory_mb": {
                "parameters": param_memory_mb,
                "forward": forward_memory_mb,
                "backward": backward_memory_mb,
                "total": param_memory_mb + backward_memory_mb,
            },
        }

    @staticmethod
    def linear_analysis(
        in_features: int, out_features: int, use_bias: bool = True
    ) -> Dict:
        """
        分析 Linear (全连接) 层的计算细节

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            use_bias: 是否使用偏置

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        # 参数量计算
        weight_params = in_features * out_features
        bias_params = out_features if use_bias else 0
        total_params = weight_params + bias_params

        # FLOPs 计算
        # y = Wx + b
        # 矩阵乘法: in_features × out_features 次乘加操作
        total_macs = in_features * out_features
        total_flops = 2 * total_macs
        if use_bias:
            total_flops += out_features

        # 内存占用 (FP32)
        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "Linear",
            "input_features": in_features,
            "output_features": out_features,
            "parameters": {
                "weight": weight_params,
                "bias": bias_params,
                "total": total_params,
            },
            "flops": {
                "macs": total_macs,
                "total": total_flops,
                "macs_readable": (
                    f"{total_macs / 1e6:.2f}M"
                    if total_macs > 1e6
                    else f"{total_macs / 1e3:.2f}K"
                ),
                "flops_readable": (
                    f"{total_flops / 1e9:.2f}G"
                    if total_flops > 1e9
                    else f"{total_flops / 1e6:.2f}M"
                ),
            },
            "memory_mb": {"parameters": param_memory_mb},
        }

    @staticmethod
    def attention_analysis(
        d_model: int, num_heads: int, seq_len: int, has_qkv_bias: bool = True
    ) -> Dict:
        """
        分析 Multi-Head Self-Attention 层的计算细节

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            seq_len: 序列长度
            has_qkv_bias: QKV投影是否使用偏置

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        # 参数量计算
        # Q, K, V 投影: 3 × (d_model × d_model)
        qkv_params = 3 * d_model * d_model
        qkv_bias = 3 * d_model if has_qkv_bias else 0

        # 输出投影: d_model × d_model
        out_params = d_model * d_model
        out_bias = d_model if has_qkv_bias else 0

        total_params = qkv_params + qkv_bias + out_params + out_bias

        # FLOPs 计算
        # 1. QKV 投影: 3 × seq_len × d_model × d_model × 2 (矩阵乘法)
        qkv_flops = 3 * seq_len * d_model * d_model * 2

        # 2. 计算注意力分数: Q @ K^T
        #    每个头: seq_len × seq_len × (d_model/num_heads)
        #    所有头: num_heads × seq_len × seq_len × (d_model/num_heads) × 2
        attn_score_flops = num_heads * seq_len * seq_len * (d_model // num_heads) * 2

        # 3. Softmax: 约 seq_len × seq_len × num_heads × 5 (exp, sum, div等)
        softmax_flops = seq_len * seq_len * num_heads * 5

        # 4. 注意力加权: attn @ V
        attn_value_flops = num_heads * seq_len * seq_len * (d_model // num_heads) * 2

        # 5. 输出投影: seq_len × d_model × d_model × 2
        out_proj_flops = seq_len * d_model * d_model * 2

        total_flops = (
            qkv_flops
            + attn_score_flops
            + softmax_flops
            + attn_value_flops
            + out_proj_flops
        )

        # 内存占用
        param_memory_mb = (total_params * 4) / (1024**2)

        # 注意力矩阵: num_heads × seq_len × seq_len
        attn_matrix_memory = (num_heads * seq_len * seq_len * 4) / (1024**2)

        return {
            "layer_type": "MultiHeadAttention",
            "d_model": d_model,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "parameters": {
                "qkv_weight": qkv_params,
                "qkv_bias": qkv_bias,
                "out_weight": out_params,
                "out_bias": out_bias,
                "total": total_params,
            },
            "flops": {
                "qkv_proj": qkv_flops,
                "attn_score": attn_score_flops,
                "softmax": softmax_flops,
                "attn_value": attn_value_flops,
                "out_proj": out_proj_flops,
                "total": total_flops,
                "flops_readable": (
                    f"{total_flops / 1e9:.2f}G"
                    if total_flops > 1e9
                    else f"{total_flops / 1e6:.2f}M"
                ),
            },
            "memory_mb": {
                "parameters": param_memory_mb,
                "attention_matrix": attn_matrix_memory,
                "total": param_memory_mb + attn_matrix_memory,
            },
        }

    @staticmethod
    def depthwise_conv2d_analysis(
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        input_shape: Tuple[int, int, int],
        use_bias: bool = True,
    ) -> Dict:
        """
        分析 Depthwise Convolution 的计算细节
        (MobileNet中使用的深度可分离卷积的第一步)

        Args:
            in_channels: 输入通道数 (也是输出通道数)
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            input_shape: (C, H, W)
            use_bias: 是否使用偏置

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        C_in, H_in, W_in = input_shape

        # 输出尺寸
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1

        # 参数量: 每个输入通道一个独立的卷积核
        weight_params = in_channels * kernel_size * kernel_size
        bias_params = in_channels if use_bias else 0
        total_params = weight_params + bias_params

        # FLOPs: 相比标准卷积大幅减少
        total_macs = in_channels * kernel_size * kernel_size * H_out * W_out
        total_flops = 2 * total_macs
        if use_bias:
            total_flops += in_channels * H_out * W_out

        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "DepthwiseConv2d",
            "input_shape": (C_in, H_in, W_in),
            "output_shape": (in_channels, H_out, W_out),
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "parameters": {
                "weight": weight_params,
                "bias": bias_params,
                "total": total_params,
            },
            "flops": {
                "macs": total_macs,
                "total": total_flops,
                "flops_readable": (
                    f"{total_flops / 1e9:.2f}G"
                    if total_flops > 1e9
                    else f"{total_flops / 1e6:.2f}M"
                ),
            },
            "memory_mb": {"parameters": param_memory_mb},
        }

    @staticmethod
    def lstm_analysis(
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> Dict:
        """
        分析 LSTM 层的计算细节

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            bias: 是否使用偏置
            bidirectional: 是否双向

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        # LSTM有4个门: input, forget, cell, output
        num_gates = 4
        num_directions = 2 if bidirectional else 1

        # 第一层参数量
        # input-to-hidden: input_size × hidden_size × 4
        # hidden-to-hidden: hidden_size × hidden_size × 4
        first_layer_params = num_gates * (
            input_size * hidden_size + hidden_size * hidden_size
        )
        if bias:
            first_layer_params += num_gates * hidden_size * 2  # ih和hh的bias

        # 其他层参数量
        other_layers_params = 0
        if num_layers > 1:
            input_size_other = hidden_size * num_directions
            other_layer_params = num_gates * (
                input_size_other * hidden_size + hidden_size * hidden_size
            )
            if bias:
                other_layer_params += num_gates * hidden_size * 2
            other_layers_params = other_layer_params * (num_layers - 1)

        # 总参数量
        params_per_direction = first_layer_params + other_layers_params
        total_params = params_per_direction * num_directions

        # FLOPs计算 (per timestep)
        # 每个时间步: 4个门 × (input_mm + hidden_mm + pointwise_ops)
        first_layer_flops = num_gates * (
            2 * input_size * hidden_size
            + 2 * hidden_size * hidden_size
            + 3 * hidden_size
        )

        other_layers_flops = 0
        if num_layers > 1:
            input_size_other = hidden_size * num_directions
            other_layer_flops = num_gates * (
                2 * input_size_other * hidden_size
                + 2 * hidden_size * hidden_size
                + 3 * hidden_size
            )
            other_layers_flops = other_layer_flops * (num_layers - 1)

        flops_per_timestep = (first_layer_flops + other_layers_flops) * num_directions

        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "LSTM",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "parameters": {
                "total": total_params,
                "per_layer": total_params // (num_layers * num_directions),
            },
            "flops": {
                "per_timestep": flops_per_timestep,
                "flops_readable": f"{flops_per_timestep / 1e6:.2f}M per timestep",
            },
            "memory_mb": {"parameters": param_memory_mb},
        }

    @staticmethod
    def layernorm_analysis(normalized_shape: int, input_shape: Tuple) -> Dict:
        """
        分析 LayerNorm 层的计算细节

        Args:
            normalized_shape: 归一化的维度
            input_shape: 输入形状

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        # 参数量: gamma 和 beta
        total_params = 2 * normalized_shape

        # FLOPs: 每个元素需要计算均值、方差、归一化、scale和shift
        total_elements = np.prod(input_shape)
        total_flops = 5 * total_elements

        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "LayerNorm",
            "normalized_shape": normalized_shape,
            "input_shape": input_shape,
            "parameters": {
                "gamma": normalized_shape,
                "beta": normalized_shape,
                "total": total_params,
            },
            "flops": {
                "total": total_flops,
                "flops_readable": (
                    f"{total_flops / 1e6:.2f}M"
                    if total_flops > 1e6
                    else f"{total_flops / 1e3:.2f}K"
                ),
            },
            "memory_mb": {"parameters": param_memory_mb},
        }

    @staticmethod
    def embedding_analysis(num_embeddings: int, embedding_dim: int) -> Dict:
        """
        分析 Embedding 层的计算细节

        Args:
            num_embeddings: 词表大小
            embedding_dim: 嵌入维度

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        # 参数量
        total_params = num_embeddings * embedding_dim

        # FLOPs: 查表操作，几乎为0
        total_flops = 0

        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "Embedding",
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "parameters": {"total": total_params},
            "flops": {"total": total_flops, "flops_readable": "~0 (lookup)"},
            "memory_mb": {"parameters": param_memory_mb},
        }

    @staticmethod
    def batchnorm2d_analysis(
        num_features: int, input_shape: Tuple[int, int, int]
    ) -> Dict:
        """
        分析 BatchNorm2d 层的计算细节

        Args:
            num_features: 通道数
            input_shape: (C, H, W)

        Returns:
            包含参数量、FLOPs、内存等信息的字典
        """
        C, H, W = input_shape

        # 参数量: gamma (scale) 和 beta (shift)
        total_params = 2 * num_features

        # FLOPs 计算
        # 每个元素: (x - mean) / sqrt(var + eps) * gamma + beta
        # = 减法 + 除法 + 乘法 + 加法 = 4 ops per element
        total_elements = C * H * W
        total_flops = 4 * total_elements

        param_memory_mb = (total_params * 4) / (1024**2)

        return {
            "layer_type": "BatchNorm2d",
            "num_features": num_features,
            "input_shape": input_shape,
            "parameters": {
                "gamma": num_features,
                "beta": num_features,
                "total": total_params,
            },
            "flops": {
                "total": total_flops,
                "flops_readable": (
                    f"{total_flops / 1e6:.2f}M"
                    if total_flops > 1e6
                    else f"{total_flops / 1e3:.2f}K"
                ),
            },
            "memory_mb": {"parameters": param_memory_mb},
        }