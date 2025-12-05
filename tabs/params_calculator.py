"""
参数量与计算量分析工具
专注于具体网络层的计算细节分析
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple


class LayerAnalyzer:
    """网络层分析器：计算参数量、FLOPs、内存占用"""
    
    @staticmethod
    def conv2d_analysis(in_channels: int, out_channels: int, kernel_size: int, 
                       stride: int, padding: int, input_shape: Tuple[int, int, int],
                       use_bias: bool = True) -> Dict:
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
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        # 前向传播激活值内存
        input_memory = C_in * H_in * W_in * 4 / (1024 ** 2)  # MB
        output_memory = out_channels * H_out * W_out * 4 / (1024 ** 2)  # MB
        forward_memory_mb = input_memory + output_memory
        
        # 反向传播需要存储输入和输出的梯度，内存翻倍
        backward_memory_mb = forward_memory_mb * 2
        
        return {
            'layer_type': 'Conv2d',
            'input_shape': (C_in, H_in, W_in),
            'output_shape': (out_channels, H_out, W_out),
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'parameters': {
                'weight': weight_params,
                'bias': bias_params,
                'total': total_params
            },
            'flops': {
                'macs': total_macs,
                'total': total_flops,
                'macs_readable': f"{total_macs / 1e6:.2f}M" if total_macs > 1e6 else f"{total_macs / 1e3:.2f}K",
                'flops_readable': f"{total_flops / 1e9:.2f}G" if total_flops > 1e9 else f"{total_flops / 1e6:.2f}M"
            },
            'memory_mb': {
                'parameters': param_memory_mb,
                'forward': forward_memory_mb,
                'backward': backward_memory_mb,
                'total': param_memory_mb + backward_memory_mb
            }
        }
    
    @staticmethod
    def linear_analysis(in_features: int, out_features: int, use_bias: bool = True) -> Dict:
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
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'Linear',
            'input_features': in_features,
            'output_features': out_features,
            'parameters': {
                'weight': weight_params,
                'bias': bias_params,
                'total': total_params
            },
            'flops': {
                'macs': total_macs,
                'total': total_flops,
                'macs_readable': f"{total_macs / 1e6:.2f}M" if total_macs > 1e6 else f"{total_macs / 1e3:.2f}K",
                'flops_readable': f"{total_flops / 1e9:.2f}G" if total_flops > 1e9 else f"{total_flops / 1e6:.2f}M"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
        }
    
    @staticmethod
    def attention_analysis(d_model: int, num_heads: int, seq_len: int, 
                          has_qkv_bias: bool = True) -> Dict:
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
        
        total_flops = qkv_flops + attn_score_flops + softmax_flops + attn_value_flops + out_proj_flops
        
        # 内存占用
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        # 注意力矩阵: num_heads × seq_len × seq_len
        attn_matrix_memory = (num_heads * seq_len * seq_len * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'MultiHeadAttention',
            'd_model': d_model,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'parameters': {
                'qkv_weight': qkv_params,
                'qkv_bias': qkv_bias,
                'out_weight': out_params,
                'out_bias': out_bias,
                'total': total_params
            },
            'flops': {
                'qkv_proj': qkv_flops,
                'attn_score': attn_score_flops,
                'softmax': softmax_flops,
                'attn_value': attn_value_flops,
                'out_proj': out_proj_flops,
                'total': total_flops,
                'flops_readable': f"{total_flops / 1e9:.2f}G" if total_flops > 1e9 else f"{total_flops / 1e6:.2f}M"
            },
            'memory_mb': {
                'parameters': param_memory_mb,
                'attention_matrix': attn_matrix_memory,
                'total': param_memory_mb + attn_matrix_memory
            }
        }
    
    @staticmethod
    def depthwise_conv2d_analysis(in_channels: int, kernel_size: int, 
                                  stride: int, padding: int, 
                                  input_shape: Tuple[int, int, int],
                                  use_bias: bool = True) -> Dict:
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
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'DepthwiseConv2d',
            'input_shape': (C_in, H_in, W_in),
            'output_shape': (in_channels, H_out, W_out),
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'parameters': {
                'weight': weight_params,
                'bias': bias_params,
                'total': total_params
            },
            'flops': {
                'macs': total_macs,
                'total': total_flops,
                'flops_readable': f"{total_flops / 1e9:.2f}G" if total_flops > 1e9 else f"{total_flops / 1e6:.2f}M"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
        }
    
    @staticmethod
    def lstm_analysis(input_size: int, hidden_size: int, num_layers: int = 1,
                     bias: bool = True, bidirectional: bool = False) -> Dict:
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
        first_layer_params = num_gates * (input_size * hidden_size + hidden_size * hidden_size)
        if bias:
            first_layer_params += num_gates * hidden_size * 2  # ih和hh的bias
        
        # 其他层参数量
        other_layers_params = 0
        if num_layers > 1:
            input_size_other = hidden_size * num_directions
            other_layer_params = num_gates * (input_size_other * hidden_size + hidden_size * hidden_size)
            if bias:
                other_layer_params += num_gates * hidden_size * 2
            other_layers_params = other_layer_params * (num_layers - 1)
        
        # 总参数量
        params_per_direction = first_layer_params + other_layers_params
        total_params = params_per_direction * num_directions
        
        # FLOPs计算 (per timestep)
        # 每个时间步: 4个门 × (input_mm + hidden_mm + pointwise_ops)
        first_layer_flops = num_gates * (2 * input_size * hidden_size + 2 * hidden_size * hidden_size + 3 * hidden_size)
        
        other_layers_flops = 0
        if num_layers > 1:
            input_size_other = hidden_size * num_directions
            other_layer_flops = num_gates * (2 * input_size_other * hidden_size + 2 * hidden_size * hidden_size + 3 * hidden_size)
            other_layers_flops = other_layer_flops * (num_layers - 1)
        
        flops_per_timestep = (first_layer_flops + other_layers_flops) * num_directions
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'LSTM',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'parameters': {
                'total': total_params,
                'per_layer': total_params // (num_layers * num_directions)
            },
            'flops': {
                'per_timestep': flops_per_timestep,
                'flops_readable': f"{flops_per_timestep / 1e6:.2f}M per timestep"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
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
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'LayerNorm',
            'normalized_shape': normalized_shape,
            'input_shape': input_shape,
            'parameters': {
                'gamma': normalized_shape,
                'beta': normalized_shape,
                'total': total_params
            },
            'flops': {
                'total': total_flops,
                'flops_readable': f"{total_flops / 1e6:.2f}M" if total_flops > 1e6 else f"{total_flops / 1e3:.2f}K"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
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
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'Embedding',
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'parameters': {
                'total': total_params
            },
            'flops': {
                'total': total_flops,
                'flops_readable': "~0 (lookup)"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
        }
    
    @staticmethod
    def batchnorm2d_analysis(num_features: int, input_shape: Tuple[int, int, int]) -> Dict:
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
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        
        return {
            'layer_type': 'BatchNorm2d',
            'num_features': num_features,
            'input_shape': input_shape,
            'parameters': {
                'gamma': num_features,
                'beta': num_features,
                'total': total_params
            },
            'flops': {
                'total': total_flops,
                'flops_readable': f"{total_flops / 1e6:.2f}M" if total_flops > 1e6 else f"{total_flops / 1e3:.2f}K"
            },
            'memory_mb': {
                'parameters': param_memory_mb
            }
        }


def params_calculator_tab():
    """参数量与FLOPs计算器标签页"""
    
    st.header("🔢 参数量与FLOPs计算器")
    
    st.markdown("""
    ### 核心功能：逐层分析网络计算细节
    
    输入网络层的配置，自动计算：
    - 📊 参数量（Params）
    - 📈 浮点运算量（FLOPs / MACs）
    - 💾 内存占用（前向/反向传播）
    - 🔍 输出特征图尺寸
    
    **与 torchinfo 的区别**：我们不仅给出数字，还展示**每个数字背后的计算公式**！
    """)
    
    # 选择层类型
    layer_type = st.selectbox(
        "选择网络层类型",
        [
            "Conv2d (标准卷积层)",
            "DepthwiseConv2d (深度可分离卷积)",
            "Linear (全连接层)",
            "MultiHeadAttention (多头注意力)",
            "LSTM (长短期记忆网络)",
            "Embedding (嵌入层)",
            "BatchNorm2d (批归一化)",
            "LayerNorm (层归一化)"
        ]
    )
    
    analyzer = LayerAnalyzer()
    
    if "Conv2d" in layer_type:
        st.markdown("### 🖼️ Conv2d 卷积层分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**输入配置**")
            C_in = st.number_input("输入通道数 (in_channels)", min_value=1, value=3, step=1)
            H_in = st.number_input("输入高度 (H)", min_value=1, value=224, step=1)
            W_in = st.number_input("输入宽度 (W)", min_value=1, value=224, step=1)
        
        with col2:
            st.markdown("**层参数配置**")
            C_out = st.number_input("输出通道数 (out_channels)", min_value=1, value=64, step=1)
            kernel_size = st.number_input("卷积核大小 (kernel_size)", min_value=1, value=7, step=1)
            stride = st.number_input("步长 (stride)", min_value=1, value=2, step=1)
            padding = st.number_input("填充 (padding)", min_value=0, value=3, step=1)
            use_bias = st.checkbox("使用偏置 (bias)", value=True)
        
        # 计算分析
        result = analyzer.conv2d_analysis(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_shape=(C_in, H_in, W_in),
            use_bias=use_bias
        )
        
        # 显示结果
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        # 输出形状
        st.markdown("#### 1️⃣ 输出特征图尺寸")
        C_out_calc, H_out, W_out = result['output_shape']
        
        st.latex(r"H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1")
        st.latex(r"W_{out} = \left\lfloor \frac{W_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("输入形状", f"[{C_in}, {H_in}, {W_in}]")
        with col2:
            st.metric("输出形状", f"[{C_out_calc}, {H_out}, {W_out}]")
        with col3:
            reduction = ((H_in * W_in) - (H_out * W_out)) / (H_in * W_in) * 100
            st.metric("空间降采样", f"{reduction:.1f}%")
        
        # 参数量
        st.markdown("#### 2️⃣ 参数量计算")
        st.latex(r"Params_{weight} = C_{out} \times C_{in} \times K_h \times K_w")
        if use_bias:
            st.latex(r"Params_{bias} = C_{out}")
            st.latex(r"Params_{total} = Params_{weight} + Params_{bias}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("权重参数", f"{result['parameters']['weight']:,}")
        with col2:
            st.metric("偏置参数", f"{result['parameters']['bias']:,}")
        with col3:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
        
        # 详细计算过程
        with st.expander("📖 查看详细计算过程"):
            st.code(f"""
计算过程：
权重参数 = {C_out} × {C_in} × {kernel_size} × {kernel_size}
        = {result['parameters']['weight']:,}

偏置参数 = {C_out if use_bias else 0}

总参数量 = {result['parameters']['weight']:,} + {result['parameters']['bias']}
        = {result['parameters']['total']:,}
            """)
        
        # FLOPs
        st.markdown("#### 3️⃣ 浮点运算量 (FLOPs)")
        st.latex(r"MACs = K_h \times K_w \times C_{in} \times C_{out} \times H_{out} \times W_{out}")
        st.latex(r"FLOPs = 2 \times MACs" + (r" + C_{out} \times H_{out} \times W_{out}" if use_bias else ""))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MACs", result['flops']['macs_readable'])
        with col2:
            st.metric("FLOPs", result['flops']['flops_readable'])
        
        with st.expander("📖 查看详细计算过程"):
            st.code(f"""
计算过程：
每个输出位置的乘加操作数 (MACs per position):
    = {kernel_size} × {kernel_size} × {C_in}
    = {kernel_size * kernel_size * C_in}

输出位置总数:
    = {C_out} × {H_out} × {W_out}
    = {C_out * H_out * W_out:,}

总 MACs:
    = {kernel_size * kernel_size * C_in} × {C_out * H_out * W_out:,}
    = {result['flops']['macs']:,}

总 FLOPs (1 MAC = 2 FLOPs):
    = 2 × {result['flops']['macs']:,}{' + ' + str(C_out * H_out * W_out) if use_bias else ''}
    = {result['flops']['total']:,}
            """)
        
        # 内存占用
        st.markdown("#### 4️⃣ 内存占用 (FP32)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("参数内存", f"{result['memory_mb']['parameters']:.4f} MB")
        with col2:
            st.metric("前向传播", f"{result['memory_mb']['forward']:.4f} MB")
        with col3:
            st.metric("反向传播", f"{result['memory_mb']['backward']:.4f} MB")
        
        # 可视化对比
        st.markdown("#### 5️⃣ 直观对比")
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=['权重参数', '偏置参数'] if use_bias else ['权重参数'],
            values=[result['parameters']['weight'], result['parameters']['bias']] if use_bias else [result['parameters']['weight']],
            hole=.3
        )])
        fig.update_layout(title_text="参数量分布", height=400)
        st.plotly_chart(fig, width='stretch')
        
    elif "Linear" in layer_type:
        st.markdown("### 🔗 Linear 全连接层分析")
        
        col1, col2 = st.columns(2)
        with col1:
            in_features = st.number_input("输入特征数 (in_features)", min_value=1, value=512, step=1)
        with col2:
            out_features = st.number_input("输出特征数 (out_features)", min_value=1, value=1000, step=1)
        
        use_bias = st.checkbox("使用偏置 (bias)", value=True, key="linear_bias")
        
        result = analyzer.linear_analysis(in_features, out_features, use_bias)
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        # 参数量
        st.markdown("#### 参数量计算")
        st.latex(r"Params_{weight} = in\_features \times out\_features")
        if use_bias:
            st.latex(r"Params_{total} = Params_{weight} + out\_features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("权重参数", f"{result['parameters']['weight']:,}")
        with col2:
            st.metric("偏置参数", f"{result['parameters']['bias']:,}")
        with col3:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
        
        # FLOPs
        st.markdown("#### 浮点运算量")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MACs", result['flops']['macs_readable'])
        with col2:
            st.metric("FLOPs", result['flops']['flops_readable'])
        
        # 警告：全连接层参数量问题
        if result['parameters']['total'] > 1e6:
            st.warning(f"""
            ⚠️ **参数量警告**
            
            该全连接层有 **{result['parameters']['total']:,}** 个参数（>{result['parameters']['total']/1e6:.1f}M）！
            
            **常见问题**：
            - 全连接层通常是网络中参数量最多的部分
            - 考虑使用 Global Average Pooling 替代
            - 或者减少输入特征数
            """)
    
    elif "BatchNorm" in layer_type:
        st.markdown("### 📏 BatchNorm2d 批归一化层分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_features = st.number_input("通道数 (num_features)", min_value=1, value=64, step=1)
        with col2:
            H = st.number_input("特征图高度 (H)", min_value=1, value=56, step=1)
        with col3:
            W = st.number_input("特征图宽度 (W)", min_value=1, value=56, step=1)
        
        result = analyzer.batchnorm2d_analysis(num_features, (num_features, H, W))
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        st.markdown("#### 参数量")
        st.latex(r"Params_{total} = 2 \times num\_features \quad (\gamma \text{ 和 } \beta)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gamma (scale)", f"{result['parameters']['gamma']}")
        with col2:
            st.metric("Beta (shift)", f"{result['parameters']['beta']}")
        
        st.info("""
        💡 **BatchNorm 参数量很小**
        
        BatchNorm 只有 2 × 通道数 个可学习参数，主要开销在于计算均值和方差。
        """)
    
    elif "DepthwiseConv2d" in layer_type:
        st.markdown("### 📱 DepthwiseConv2d 深度可分离卷积分析")
        
        st.info("""
        💡 **MobileNet的核心技术**
        
        深度可分离卷积将标准卷积分解为：
        1. Depthwise Convolution (逐通道卷积)
        2. Pointwise Convolution (1×1卷积)
        
        大幅减少参数量和计算量！
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**输入配置**")
            C_in = st.number_input("输入通道数", min_value=1, value=64, step=1, key="dw_cin")
            H_in = st.number_input("输入高度", min_value=1, value=56, step=1, key="dw_hin")
            W_in = st.number_input("输入宽度", min_value=1, value=56, step=1, key="dw_win")
        
        with col2:
            st.markdown("**层参数配置**")
            kernel_size = st.number_input("卷积核大小", min_value=1, value=3, step=1, key="dw_kernel")
            stride = st.number_input("步长", min_value=1, value=1, step=1, key="dw_stride")
            padding = st.number_input("填充", min_value=0, value=1, step=1, key="dw_padding")
            use_bias = st.checkbox("使用偏置", value=True, key="dw_bias")
        
        # 计算Depthwise卷积
        result_dw = analyzer.depthwise_conv2d_analysis(
            C_in, kernel_size, stride, padding, (C_in, H_in, W_in), use_bias
        )
        
        # 计算标准卷积作为对比
        result_std = analyzer.conv2d_analysis(
            C_in, C_in, kernel_size, stride, padding, (C_in, H_in, W_in), use_bias
        )
        
        st.markdown("---")
        st.markdown("### 📊 对比分析：Depthwise vs 标准卷积")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**参数量对比**")
            dw_params = result_dw['parameters']['total']
            std_params = result_std['parameters']['total']
            reduction = (1 - dw_params / std_params) * 100
            
            st.metric("Depthwise", f"{dw_params:,}")
            st.metric("标准卷积", f"{std_params:,}")
            st.metric("参数减少", f"{reduction:.1f}%", delta=f"-{std_params - dw_params:,}")
        
        with col2:
            st.markdown("**FLOPs对比**")
            dw_flops = result_dw['flops']['total']
            std_flops = result_std['flops']['total']
            flops_reduction = (1 - dw_flops / std_flops) * 100
            
            st.metric("Depthwise", result_dw['flops']['flops_readable'])
            st.metric("标准卷积", result_std['flops']['flops_readable'])
            st.metric("计算减少", f"{flops_reduction:.1f}%")
        
        with col3:
            st.markdown("**输出形状**")
            st.metric("输入", f"{(C_in, H_in, W_in)}")
            st.metric("输出", f"{result_dw['output_shape']}")
        
        # 详细说明
        with st.expander("📖 为什么参数量大幅减少？"):
            st.markdown(f"""
            **标准卷积参数量**:
            ```
            C_out × C_in × K × K
            = {C_in} × {C_in} × {kernel_size} × {kernel_size}
            = {std_params:,}
            ```
            
            **Depthwise卷积参数量**:
            ```
            C_in × K × K  (每个通道独立的卷积核)
            = {C_in} × {kernel_size} × {kernel_size}
            = {dw_params:,}
            ```
            
            **减少因子**: 约 **1/{C_in}** = 1/{C_in} ≈ {std_params/dw_params:.1f}x
            """)
    
    elif "MultiHeadAttention" in layer_type:
        st.markdown("### 🎯 Multi-Head Self-Attention 分析")
        
        st.info("""
        💡 **Transformer的核心组件**
        
        多头注意力机制可以让模型关注输入的不同表示子空间。
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            d_model = st.number_input("模型维度 (d_model)", min_value=64, max_value=2048, value=512, step=64, key="attn_d")
        with col2:
            num_heads = st.number_input("注意力头数", min_value=1, max_value=32, value=8, step=1, key="attn_heads")
        with col3:
            seq_len = st.number_input("序列长度", min_value=1, max_value=2048, value=128, step=1, key="attn_seq")
        
        has_qkv_bias = st.checkbox("QKV投影使用偏置", value=True, key="attn_bias")
        
        # 检查d_model是否能被num_heads整除
        if d_model % num_heads != 0:
            st.error(f"⚠️ d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除！")
            st.stop()
        
        result = analyzer.attention_analysis(d_model, num_heads, seq_len, has_qkv_bias)
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        # 基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("每个头的维度", f"{d_model // num_heads}")
        with col2:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
        with col3:
            st.metric("总FLOPs", result['flops']['flops_readable'])
        
        # 参数量分解
        st.markdown("#### 参数量分解")
        
        params_breakdown = pd.DataFrame({
            '组件': ['Q投影', 'K投影', 'V投影', 'QKV偏置', '输出投影', '输出偏置'],
            '参数量': [
                d_model * d_model,
                d_model * d_model,
                d_model * d_model,
                3 * d_model if has_qkv_bias else 0,
                d_model * d_model,
                d_model if has_qkv_bias else 0
            ],
            '占比': [
                f"{d_model * d_model / result['parameters']['total'] * 100:.1f}%",
                f"{d_model * d_model / result['parameters']['total'] * 100:.1f}%",
                f"{d_model * d_model / result['parameters']['total'] * 100:.1f}%",
                f"{(3 * d_model if has_qkv_bias else 0) / result['parameters']['total'] * 100:.1f}%",
                f"{d_model * d_model / result['parameters']['total'] * 100:.1f}%",
                f"{(d_model if has_qkv_bias else 0) / result['parameters']['total'] * 100:.1f}%"
            ]
        })
        
        st.dataframe(params_breakdown, use_container_width=True)
        
        # FLOPs分解
        st.markdown("#### FLOPs分解")
        
        flops_breakdown = pd.DataFrame({
            '操作': ['QKV投影', '注意力分数(Q@K^T)', 'Softmax', '注意力加权(Attn@V)', '输出投影'],
            'FLOPs': [
                result['flops']['qkv_proj'],
                result['flops']['attn_score'],
                result['flops']['softmax'],
                result['flops']['attn_value'],
                result['flops']['out_proj']
            ],
            '占比': [
                f"{result['flops']['qkv_proj'] / result['flops']['total'] * 100:.1f}%",
                f"{result['flops']['attn_score'] / result['flops']['total'] * 100:.1f}%",
                f"{result['flops']['softmax'] / result['flops']['total'] * 100:.1f}%",
                f"{result['flops']['attn_value'] / result['flops']['total'] * 100:.1f}%",
                f"{result['flops']['out_proj'] / result['flops']['total'] * 100:.1f}%"
            ]
        })
        
        st.dataframe(flops_breakdown, use_container_width=True)
        
        # 可视化
        fig = go.Figure(data=[go.Pie(
            labels=flops_breakdown['操作'],
            values=flops_breakdown['FLOPs'],
            hole=.3
        )])
        fig.update_layout(title="FLOPs分布", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 复杂度分析
        st.markdown("#### 🔬 复杂度分析")
        
        st.markdown(f"""
        **注意力机制的二次复杂度**:
        - 计算注意力矩阵: O(seq_len²) = O({seq_len}²) = {seq_len**2:,} 个位置
        - 当序列长度增加时，计算量和内存占用呈**平方增长**
        - 内存占用（注意力矩阵）: {result['memory_mb']['attention_matrix']:.4f} MB
        
        **优化建议**:
        - 使用稀疏注意力 (Sparse Attention)
        - 使用线性注意力 (Linear Attention)
        - 使用局部窗口注意力 (如 Swin Transformer)
        """)
    
    elif "LSTM" in layer_type:
        st.markdown("### 🔄 LSTM 长短期记忆网络分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**网络配置**")
            input_size = st.number_input("输入维度", min_value=1, value=128, step=1, key="lstm_in")
            hidden_size = st.number_input("隐藏层维度", min_value=1, value=256, step=1, key="lstm_hidden")
        
        with col2:
            st.markdown("**层配置**")
            num_layers = st.number_input("层数", min_value=1, max_value=10, value=2, step=1, key="lstm_layers")
            bidirectional = st.checkbox("双向LSTM", value=False, key="lstm_bi")
            use_bias = st.checkbox("使用偏置", value=True, key="lstm_bias")
        
        result = analyzer.lstm_analysis(input_size, hidden_size, num_layers, use_bias, bidirectional)
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
            st.metric("每层参数量", f"{result['parameters']['per_layer']:,}")
        
        with col2:
            st.metric("FLOPs/时间步", result['flops']['flops_readable'])
            seq_length = st.number_input("序列长度", min_value=1, value=50, step=1, key="lstm_seqlen")
            total_flops = result['flops']['per_timestep'] * seq_length
            st.metric("总FLOPs", f"{total_flops/1e9:.2f}G" if total_flops > 1e9 else f"{total_flops/1e6:.2f}M")
        
        with col3:
            st.metric("参数内存", f"{result['memory_mb']['parameters']:.2f} MB")
            direction_text = "双向" if bidirectional else "单向"
            st.metric("方向", direction_text)
        
        # LSTM结构说明
        st.markdown("#### 🧠 LSTM内部结构")
        
        st.markdown("""
        LSTM有**4个门**，每个门都需要权重矩阵：
        1. **输入门 (Input Gate)**: 决定新信息的重要性
        2. **遗忘门 (Forget Gate)**: 决定丢弃哪些信息
        3. **细胞门 (Cell Gate)**: 创建新的候选值
        4. **输出门 (Output Gate)**: 决定输出什么
        """)
        
        # 参数量公式
        st.markdown("#### 📐 参数量计算公式")
        
        st.latex(r"Params_{layer1} = 4 \times (input\_size \times hidden\_size + hidden\_size^2)")
        
        if num_layers > 1:
            input_size_other = hidden_size * (2 if bidirectional else 1)
            st.latex(r"Params_{other} = 4 \times (hidden\_size \times num\_directions \times hidden\_size + hidden\_size^2)")
        
        with st.expander("📖 查看详细计算"):
            st.code(f"""
第一层参数量:
    input-to-hidden: 4 × {input_size} × {hidden_size} = {4 * input_size * hidden_size:,}
    hidden-to-hidden: 4 × {hidden_size} × {hidden_size} = {4 * hidden_size * hidden_size:,}
    偏置: 4 × {hidden_size} × 2 = {4 * hidden_size * 2:,}
    小计: {result['parameters']['per_layer']:,}

{'其他' + str(num_layers-1) + '层参数量:' if num_layers > 1 else ''}
{'    ' + str((num_layers-1) * result['parameters']['per_layer']) + ' (每层相同)' if num_layers > 1 else ''}

总参数量: {result['parameters']['total']:,}
            """)
        
        # 与GRU对比
        if st.checkbox("与GRU对比", key="lstm_compare_gru"):
            gru_params = num_layers * 3 * (input_size * hidden_size + hidden_size * hidden_size)
            if bidirectional:
                gru_params *= 2
            
            st.markdown("#### 🆚 LSTM vs GRU")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("LSTM参数量", f"{result['parameters']['total']:,}")
            with col2:
                st.metric("GRU参数量 (估算)", f"{gru_params:,}")
            
            st.info("""
            💡 **LSTM vs GRU**
            
            - **LSTM**: 4个门，更强的表达能力，但参数更多
            - **GRU**: 3个门，参数约为LSTM的75%，训练更快
            - 在大多数任务上性能相近，GRU更轻量
            """)
    
    elif "Embedding" in layer_type:
        st.markdown("### 📚 Embedding 嵌入层分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_embeddings = st.number_input(
                "词表大小 (num_embeddings)",
                min_value=100,
                max_value=1000000,
                value=30000,
                step=1000,
                key="emb_vocab"
            )
        
        with col2:
            embedding_dim = st.number_input(
                "嵌入维度 (embedding_dim)",
                min_value=16,
                max_value=2048,
                value=512,
                step=64,
                key="emb_dim"
            )
        
        result = analyzer.embedding_analysis(num_embeddings, embedding_dim)
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
            readable = f"{result['parameters']['total']/1e6:.2f}M" if result['parameters']['total'] > 1e6 else f"{result['parameters']['total']/1e3:.2f}K"
            st.metric("可读格式", readable)
        
        with col2:
            st.metric("参数内存", f"{result['memory_mb']['parameters']:.2f} MB")
        
        with col3:
            st.metric("FLOPs", result['flops']['flops_readable'])
        
        # 参数量公式
        st.markdown("#### 📐 参数量计算")
        st.latex(r"Params = num\_embeddings \times embedding\_dim")
        st.code(f"""
计算过程:
{num_embeddings:,} × {embedding_dim} = {result['parameters']['total']:,}
        """)
        
        # 常见词表大小参考
        st.markdown("#### 📚 常见词表大小参考")
        
        vocab_sizes = pd.DataFrame({
            '模型/场景': ['BERT-base', 'GPT-2', 'T5', 'LLaMA', '中文模型', '多语言模型'],
            '词表大小': ['30,522', '50,257', '32,128', '32,000', '21,128', '250,000+'],
            '嵌入维度': [768, 768, 512, 4096, 768, 1024]
        })
        
        st.dataframe(vocab_sizes, use_container_width=True)
        
        # 警告
        if result['parameters']['total'] > 10e6:
            st.warning(f"""
            ⚠️ **参数量警告**
            
            嵌入层有 **{result['parameters']['total']/1e6:.1f}M** 参数！
            
            **优化建议**:
            - 使用子词分词（BPE, WordPiece）减小词表
            - 使用哈希技巧（Hash Trick）
            - 权重共享（如输入输出嵌入共享）
            - 使用更小的嵌入维度
            """)
    
    elif "LayerNorm" in layer_type:
        st.markdown("### 📏 LayerNorm 层归一化分析")
        
        st.info("""
        💡 **Transformer中的标准归一化方式**
        
        LayerNorm对每个样本的特征维度进行归一化，与BatchNorm不同，不依赖batch统计。
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            normalized_shape = st.number_input(
                "归一化维度 (normalized_shape)",
                min_value=1,
                value=512,
                step=64,
                key="ln_shape"
            )
        
        with col2:
            batch_size = st.number_input("批次大小", min_value=1, value=32, step=1, key="ln_batch")
            seq_len = st.number_input("序列长度", min_value=1, value=128, step=1, key="ln_seq")
        
        input_shape = (batch_size, seq_len, normalized_shape)
        
        result = analyzer.layernorm_analysis(normalized_shape, input_shape)
        
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总参数量", f"{result['parameters']['total']:,}")
        
        with col2:
            st.metric("FLOPs", result['flops']['flops_readable'])
        
        with col3:
            st.metric("参数内存", f"{result['memory_mb']['parameters']:.4f} MB")
        
        # 参数说明
        st.markdown("#### 参数构成")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gamma (缩放)", result['parameters']['gamma'])
        with col2:
            st.metric("Beta (平移)", result['parameters']['beta'])
        
        # LayerNorm vs BatchNorm
        st.markdown("#### 🆚 LayerNorm vs BatchNorm")
        
        comparison = pd.DataFrame({
            '特性': ['归一化维度', 'Batch依赖', '适用场景', '参数量', '训练/推理差异'],
            'LayerNorm': [
                '特征维度 (Feature)',
                '否',
                'Transformer, RNN',
                f'{normalized_shape * 2}',
                '无差异'
            ],
            'BatchNorm': [
                '批次维度 (Batch)',
                '是',
                'CNN',
                f'{normalized_shape * 2}',
                '需要running_mean/var'
            ]
        })
        
        st.dataframe(comparison, use_container_width=True)
        
        st.info("""
        💡 **为什么Transformer用LayerNorm？**
        
        - 不依赖批次大小，适合小batch训练
        - 训练和推理行为一致
        - 对序列长度变化不敏感
        - 更适合NLP任务的特征分布
        """)
    
    # 添加完整网络分析入口
    st.markdown("---")
    st.markdown("## 🏗️ 完整网络分析")
    
    if st.button("切换到完整网络分析模式", use_container_width=True):
        st.session_state.calc_mode = "network"
        st.rerun()
    
    # 显示完整网络分析
    if st.session_state.get('calc_mode') == 'network':
        _full_network_analysis()


def _full_network_analysis():
    """完整网络分析模式"""
    st.markdown("---")
    st.markdown("## 🏗️ 完整网络参数分析")
    
    st.markdown("""
    选择预定义网络或自定义网络架构，生成详细的参数/FLOPs报告。
    """)
    
    # 网络选择
    network_mode = st.radio(
        "选择模式",
        ["预定义网络", "自定义网络"],
        horizontal=True,
        key="network_mode"
    )
    
    if network_mode == "预定义网络":
        _predefined_network_analysis()
    else:
        _custom_network_analysis()
    
    # 返回单层分析
    if st.button("返回单层分析", use_container_width=True):
        st.session_state.calc_mode = "single"
        st.rerun()


def _predefined_network_analysis():
    """预定义网络分析"""
    st.markdown("### 📦 预定义网络架构")
    
    network_name = st.selectbox(
        "选择网络",
        [
            "ResNet-18 (CNN)",
            "ResNet-50 (CNN)",
            "VGG-16 (CNN)",
            "MobileNetV2 (轻量级CNN)",
            "BERT-base (Transformer)",
            "GPT-2 small (Transformer)",
            "ViT-Base (Vision Transformer)"
        ],
        key="predefined_network"
    )
    
    # 输入尺寸
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("批次大小", min_value=1, value=1, step=1, key="batch_size")
    with col2:
        input_size = st.selectbox("输入尺寸", [224, 256, 384, 512], index=0, key="input_size")
    
    # 获取网络架构
    network_config = _get_network_config(network_name, input_size)
    
    # 计算总体统计
    total_params = 0
    total_flops = 0
    total_memory = 0
    
    layers_data = []
    
    for layer_info in network_config:
        total_params += layer_info['params']
        total_flops += layer_info['flops']
        total_memory += layer_info.get('memory', 0)
        layers_data.append(layer_info)
    
    # 显示总体统计
    st.markdown("---")
    st.markdown("### 📊 网络总体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总参数量",
            f"{total_params/1e6:.2f}M",
            help="网络中所有可学习参数的总数"
        )
    
    with col2:
        st.metric(
            "总FLOPs",
            f"{total_flops/1e9:.2f}G",
            help="单次前向传播的浮点运算次数"
        )
    
    with col3:
        st.metric(
            "参数内存",
            f"{total_params*4/1024/1024:.2f}MB",
            help="存储所有参数需要的内存（FP32）"
        )
    
    with col4:
        st.metric(
            "激活内存",
            f"{total_memory:.2f}MB",
            help="前向传播激活值占用的内存"
        )
    
    # 逐层详细信息
    st.markdown("---")
    st.markdown("### 📋 逐层详细分析")
    
    # 创建数据表格
    df = pd.DataFrame(layers_data)
    
    # 格式化显示
    df['params_readable'] = df['params'].apply(lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.2f}K")
    df['flops_readable'] = df['flops'].apply(lambda x: f"{x/1e9:.2f}G" if x > 1e9 else f"{x/1e6:.2f}M")
    df['output_shape_str'] = df['output_shape'].apply(lambda x: f"{x}")
    
    display_df = df[['layer_name', 'layer_type', 'output_shape_str', 'params_readable', 'flops_readable']]
    display_df.columns = ['层名称', '层类型', '输出形状', '参数量', 'FLOPs']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # 可视化
    st.markdown("---")
    st.markdown("### 📈 可视化分析")
    
    # 参数量分布
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure(data=[go.Bar(
            x=df['layer_name'],
            y=df['params'],
            marker_color='lightblue'
        )])
        fig1.update_layout(
            title="各层参数量分布",
            xaxis_title="层名称",
            yaxis_title="参数量",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(data=[go.Bar(
            x=df['layer_name'],
            y=df['flops'],
            marker_color='lightcoral'
        )])
        fig2.update_layout(
            title="各层FLOPs分布",
            xaxis_title="层名称",
            yaxis_title="FLOPs",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 饼图：参数量占比
    st.markdown("#### 参数量占比")
    
    # 按层类型聚合
    layer_type_params = df.groupby('layer_type')['params'].sum()
    
    fig3 = go.Figure(data=[go.Pie(
        labels=layer_type_params.index,
        values=layer_type_params.values,
        hole=.3
    )])
    fig3.update_layout(title="按层类型的参数量分布", height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # 生成报告
    st.markdown("---")
    st.markdown("### 📄 生成详细报告")
    
    if st.button("生成Markdown报告", use_container_width=True):
        report = _generate_network_report(network_name, input_size, batch_size, layers_data, total_params, total_flops)
        st.code(report, language="markdown")
        st.download_button(
            "下载报告",
            report,
            file_name=f"{network_name}_analysis.md",
            mime="text/markdown"
        )


def _custom_network_analysis():
    """自定义网络分析"""
    st.markdown("### 🛠️ 自定义网络架构")
    
    st.markdown("""
    **快速构建自定义网络并分析参数量。**
    
    在下方添加层，我们会自动计算参数量和FLOPs。
    """)
    
    # 初始化session state
    if 'custom_layers' not in st.session_state:
        st.session_state.custom_layers = []
    
    # 输入配置
    col1, col2, col3 = st.columns(3)
    with col1:
        input_channels = st.number_input("输入通道数", min_value=1, value=3, step=1, key="custom_input_c")
    with col2:
        input_height = st.number_input("输入高度", min_value=1, value=224, step=1, key="custom_input_h")
    with col3:
        input_width = st.number_input("输入宽度", min_value=1, value=224, step=1, key="custom_input_w")
    
    current_shape = (input_channels, input_height, input_width)
    
    # 添加层
    st.markdown("---")
    st.markdown("#### 添加网络层")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        layer_to_add = st.selectbox(
            "选择层类型",
            ["Conv2d", "Linear", "MaxPool2d", "BatchNorm2d", "ReLU"],
            key="layer_to_add"
        )
    
    with col2:
        if st.button("添加层", use_container_width=True):
            st.session_state.custom_layers.append({'type': layer_to_add, 'params': {}})
            st.rerun()
    
    # 配置每一层
    if st.session_state.custom_layers:
        st.markdown("#### 配置网络层")
        
        analyzer = LayerAnalyzer()
        total_params = 0
        total_flops = 0
        
        for idx, layer in enumerate(st.session_state.custom_layers):
            with st.expander(f"第 {idx+1} 层: {layer['type']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if layer['type'] == 'Conv2d':
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            out_channels = st.number_input("输出通道", min_value=1, value=64, key=f"conv_out_{idx}")
                        with col_b:
                            kernel = st.number_input("卷积核", min_value=1, value=3, key=f"conv_k_{idx}")
                        with col_c:
                            stride = st.number_input("步长", min_value=1, value=1, key=f"conv_s_{idx}")
                        with col_d:
                            padding = st.number_input("填充", min_value=0, value=1, key=f"conv_p_{idx}")
                        
                        result = analyzer.conv2d_analysis(
                            current_shape[0], out_channels, kernel, stride, padding, current_shape
                        )
                        current_shape = result['output_shape']
                        total_params += result['parameters']['total']
                        total_flops += result['flops']['total']
                        
                        st.write(f"输出形状: {current_shape}")
                        st.write(f"参数量: {result['parameters']['total']:,}")
                        st.write(f"FLOPs: {result['flops']['flops_readable']}")
                    
                    elif layer['type'] == 'Linear':
                        out_features = st.number_input("输出特征数", min_value=1, value=1000, key=f"linear_out_{idx}")
                        
                        # 如果前面是Conv，需要flatten
                        if len(current_shape) == 3:
                            in_features = current_shape[0] * current_shape[1] * current_shape[2]
                            st.info(f"自动展平: {current_shape} → {in_features}")
                        else:
                            in_features = current_shape[0]
                        
                        result = analyzer.linear_analysis(in_features, out_features)
                        current_shape = (out_features,)
                        total_params += result['parameters']['total']
                        total_flops += result['flops']['total']
                        
                        st.write(f"输出形状: {current_shape}")
                        st.write(f"参数量: {result['parameters']['total']:,}")
                        st.write(f"FLOPs: {result['flops']['flops_readable']}")
                    
                    elif layer['type'] == 'MaxPool2d':
                        col_a, col_b = st.columns(2)
                        with col_a:
                            pool_kernel = st.number_input("池化核", min_value=1, value=2, key=f"pool_k_{idx}")
                        with col_b:
                            pool_stride = st.number_input("池化步长", min_value=1, value=2, key=f"pool_s_{idx}")
                        
                        if len(current_shape) == 3:
                            new_h = (current_shape[1] - pool_kernel) // pool_stride + 1
                            new_w = (current_shape[2] - pool_kernel) // pool_stride + 1
                            current_shape = (current_shape[0], new_h, new_w)
                        
                        st.write(f"输出形状: {current_shape}")
                        st.write(f"参数量: 0 (无可学习参数)")
                    
                    elif layer['type'] == 'BatchNorm2d':
                        if len(current_shape) == 3:
                            result = analyzer.batchnorm2d_analysis(current_shape[0], current_shape)
                            total_params += result['parameters']['total']
                            total_flops += result['flops']['total']
                            
                            st.write(f"输出形状: {current_shape}")
                            st.write(f"参数量: {result['parameters']['total']:,}")
                    
                    elif layer['type'] == 'ReLU':
                        st.write(f"输出形状: {current_shape}")
                        st.write(f"参数量: 0 (激活函数无参数)")
                
                with col2:
                    if st.button("删除", key=f"del_{idx}", use_container_width=True):
                        st.session_state.custom_layers.pop(idx)
                        st.rerun()
        
        # 总体统计
        st.markdown("---")
        st.markdown("### 📊 网络总体统计")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总层数", len(st.session_state.custom_layers))
        with col2:
            st.metric("总参数量", f"{total_params/1e6:.2f}M" if total_params > 1e6 else f"{total_params:,}")
        with col3:
            st.metric("总FLOPs", f"{total_flops/1e9:.2f}G" if total_flops > 1e9 else f"{total_flops/1e6:.2f}M")
        
        # 清空按钮
        if st.button("清空所有层", use_container_width=True):
            st.session_state.custom_layers = []
            st.rerun()


def _get_network_config(network_name: str, input_size: int) -> List[Dict]:
    """获取预定义网络的配置"""
    
    if "ResNet-18" in network_name:
        return [
            {'layer_name': 'conv1', 'layer_type': 'Conv2d', 'output_shape': (64, input_size//2, input_size//2), 
             'params': 9408, 'flops': 118013952},
            {'layer_name': 'layer1', 'layer_type': 'ResBlock', 'output_shape': (64, input_size//2, input_size//2), 
             'params': 147968, 'flops': 924844032},
            {'layer_name': 'layer2', 'layer_type': 'ResBlock', 'output_shape': (128, input_size//4, input_size//4), 
             'params': 525568, 'flops': 924844032},
            {'layer_name': 'layer3', 'layer_type': 'ResBlock', 'output_shape': (256, input_size//8, input_size//8), 
             'params': 2099712, 'flops': 924844032},
            {'layer_name': 'layer4', 'layer_type': 'ResBlock', 'output_shape': (512, input_size//16, input_size//16), 
             'params': 8394752, 'flops': 924844032},
            {'layer_name': 'fc', 'layer_type': 'Linear', 'output_shape': (1000,), 
             'params': 513000, 'flops': 1024000},
        ]
    
    elif "VGG-16" in network_name:
        return [
            {'layer_name': 'conv1_1', 'layer_type': 'Conv2d', 'output_shape': (64, input_size, input_size), 
             'params': 1792, 'flops': 86704128},
            {'layer_name': 'conv1_2', 'layer_type': 'Conv2d', 'output_shape': (64, input_size, input_size), 
             'params': 36928, 'flops': 1849688064},
            {'layer_name': 'pool1', 'layer_type': 'MaxPool2d', 'output_shape': (64, input_size//2, input_size//2), 
             'params': 0, 'flops': 0},
            {'layer_name': 'conv2_1', 'layer_type': 'Conv2d', 'output_shape': (128, input_size//2, input_size//2), 
             'params': 73856, 'flops': 924844032},
            {'layer_name': 'conv2_2', 'layer_type': 'Conv2d', 'output_shape': (128, input_size//2, input_size//2), 
             'params': 147584, 'flops': 1849688064},
            {'layer_name': 'fc', 'layer_type': 'Linear', 'output_shape': (1000,), 
             'params': 4096000, 'flops': 8192000},
        ]
    
    elif "MobileNetV2" in network_name:
        return [
            {'layer_name': 'conv1', 'layer_type': 'Conv2d', 'output_shape': (32, input_size//2, input_size//2), 
             'params': 864, 'flops': 10838016},
            {'layer_name': 'bottleneck1', 'layer_type': 'InvertedResidual', 'output_shape': (16, input_size//2, input_size//2), 
             'params': 896, 'flops': 11239424},
            {'layer_name': 'bottleneck2', 'layer_type': 'InvertedResidual', 'output_shape': (24, input_size//4, input_size//4), 
             'params': 5136, 'flops': 40140800},
            {'layer_name': 'bottleneck3', 'layer_type': 'InvertedResidual', 'output_shape': (32, input_size//8, input_size//8), 
             'params': 8832, 'flops': 34406400},
            {'layer_name': 'bottleneck4', 'layer_type': 'InvertedResidual', 'output_shape': (64, input_size//16, input_size//16), 
             'params': 25728, 'flops': 50135040},
            {'layer_name': 'bottleneck5', 'layer_type': 'InvertedResidual', 'output_shape': (96, input_size//16, input_size//16), 
             'params': 66624, 'flops': 129957888},
            {'layer_name': 'bottleneck6', 'layer_type': 'InvertedResidual', 'output_shape': (160, input_size//32, input_size//32), 
             'params': 118272, 'flops': 91570176},
            {'layer_name': 'bottleneck7', 'layer_type': 'InvertedResidual', 'output_shape': (320, input_size//32, input_size//32), 
             'params': 155264, 'flops': 120197120},
            {'layer_name': 'conv_last', 'layer_type': 'Conv2d', 'output_shape': (1280, input_size//32, input_size//32), 
             'params': 409600, 'flops': 200704000},
            {'layer_name': 'classifier', 'layer_type': 'Linear', 'output_shape': (1000,), 
             'params': 1281000, 'flops': 2560000},
        ]
    
    elif "BERT-base" in network_name:
        # BERT-base: 12层Transformer, d_model=768, num_heads=12
        d_model = 768
        num_heads = 12
        seq_len = 512
        vocab_size = 30522
        
        layers = []
        
        # Embedding层
        layers.append({
            'layer_name': 'token_embeddings',
            'layer_type': 'Embedding',
            'output_shape': (seq_len, d_model),
            'params': vocab_size * d_model,  # 23,440,896
            'flops': 0,
            'memory': 0
        })
        
        layers.append({
            'layer_name': 'position_embeddings',
            'layer_type': 'Embedding',
            'output_shape': (seq_len, d_model),
            'params': 512 * d_model,  # 393,216
            'flops': 0,
            'memory': 0
        })
        
        # 12个Transformer层
        for i in range(12):
            # Multi-Head Attention
            attn_params = 4 * d_model * d_model + 4 * d_model  # Q,K,V,O + bias
            attn_flops = 6 * seq_len * d_model * d_model + 2 * num_heads * seq_len * seq_len * (d_model // num_heads) * 2
            
            layers.append({
                'layer_name': f'layer{i}_attention',
                'layer_type': 'MultiHeadAttention',
                'output_shape': (seq_len, d_model),
                'params': attn_params,  # 2,362,368
                'flops': attn_flops,
                'memory': 0
            })
            
            # LayerNorm
            layers.append({
                'layer_name': f'layer{i}_ln1',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,  # 1,536
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
            
            # Feed Forward (两层Linear)
            ffn_hidden = d_model * 4  # 3072
            ffn_params = d_model * ffn_hidden + ffn_hidden + ffn_hidden * d_model + d_model
            ffn_flops = 2 * seq_len * (d_model * ffn_hidden + ffn_hidden * d_model)
            
            layers.append({
                'layer_name': f'layer{i}_ffn',
                'layer_type': 'FeedForward',
                'output_shape': (seq_len, d_model),
                'params': ffn_params,  # 4,722,432
                'flops': ffn_flops,
                'memory': 0
            })
            
            # LayerNorm
            layers.append({
                'layer_name': f'layer{i}_ln2',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,  # 1,536
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
        
        # Pooler
        layers.append({
            'layer_name': 'pooler',
            'layer_type': 'Linear',
            'output_shape': (d_model,),
            'params': d_model * d_model + d_model,  # 590,592
            'flops': 2 * d_model * d_model,
            'memory': 0
        })
        
        return layers
    
    elif "GPT-2" in network_name:
        # GPT-2 small: 12层Transformer, d_model=768, num_heads=12
        d_model = 768
        num_heads = 12
        seq_len = 1024
        vocab_size = 50257
        
        layers = []
        
        # Token Embedding
        layers.append({
            'layer_name': 'token_embeddings',
            'layer_type': 'Embedding',
            'output_shape': (seq_len, d_model),
            'params': vocab_size * d_model,  # 38,597,376
            'flops': 0,
            'memory': 0
        })
        
        # Position Embedding
        layers.append({
            'layer_name': 'position_embeddings',
            'layer_type': 'Embedding',
            'output_shape': (seq_len, d_model),
            'params': seq_len * d_model,  # 786,432
            'flops': 0,
            'memory': 0
        })
        
        # 12个Transformer块
        for i in range(12):
            # LayerNorm 1
            layers.append({
                'layer_name': f'layer{i}_ln1',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
            
            # Causal Self-Attention
            attn_params = 4 * d_model * d_model + 4 * d_model
            attn_flops = 6 * seq_len * d_model * d_model + 2 * num_heads * seq_len * seq_len * (d_model // num_heads) * 2
            
            layers.append({
                'layer_name': f'layer{i}_attn',
                'layer_type': 'CausalAttention',
                'output_shape': (seq_len, d_model),
                'params': attn_params,
                'flops': attn_flops,
                'memory': 0
            })
            
            # LayerNorm 2
            layers.append({
                'layer_name': f'layer{i}_ln2',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
            
            # MLP
            ffn_hidden = d_model * 4
            ffn_params = d_model * ffn_hidden + ffn_hidden + ffn_hidden * d_model + d_model
            ffn_flops = 2 * seq_len * (d_model * ffn_hidden + ffn_hidden * d_model)
            
            layers.append({
                'layer_name': f'layer{i}_mlp',
                'layer_type': 'MLP',
                'output_shape': (seq_len, d_model),
                'params': ffn_params,
                'flops': ffn_flops,
                'memory': 0
            })
        
        # Final LayerNorm
        layers.append({
            'layer_name': 'ln_f',
            'layer_type': 'LayerNorm',
            'output_shape': (seq_len, d_model),
            'params': 2 * d_model,
            'flops': 5 * seq_len * d_model,
            'memory': 0
        })
        
        # Language Model Head (共享embedding权重，所以参数为0)
        layers.append({
            'layer_name': 'lm_head',
            'layer_type': 'Linear',
            'output_shape': (vocab_size,),
            'params': 0,  # 权重共享
            'flops': 2 * seq_len * vocab_size * d_model,
            'memory': 0
        })
        
        return layers
    
    elif "ViT-Base" in network_name:
        # Vision Transformer Base: patch_size=16, d_model=768, num_heads=12, 12层
        patch_size = 16
        d_model = 768
        num_heads = 12
        num_patches = (input_size // patch_size) ** 2  # 196 for 224x224
        seq_len = num_patches + 1  # +1 for class token
        
        layers = []
        
        # Patch Embedding
        layers.append({
            'layer_name': 'patch_embed',
            'layer_type': 'Conv2d',
            'output_shape': (d_model, input_size//patch_size, input_size//patch_size),
            'params': 3 * patch_size * patch_size * d_model + d_model,  # 590,592
            'flops': 3 * patch_size * patch_size * d_model * num_patches * 2,
            'memory': 0
        })
        
        # Position Embedding
        layers.append({
            'layer_name': 'pos_embed',
            'layer_type': 'Embedding',
            'output_shape': (seq_len, d_model),
            'params': seq_len * d_model,  # 151,296 for 224x224
            'flops': 0,
            'memory': 0
        })
        
        # 12个Transformer编码器层
        for i in range(12):
            # LayerNorm + Attention
            layers.append({
                'layer_name': f'block{i}_ln1',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
            
            attn_params = 4 * d_model * d_model + 4 * d_model
            attn_flops = 6 * seq_len * d_model * d_model + 2 * num_heads * seq_len * seq_len * (d_model // num_heads) * 2
            
            layers.append({
                'layer_name': f'block{i}_attn',
                'layer_type': 'MultiHeadAttention',
                'output_shape': (seq_len, d_model),
                'params': attn_params,
                'flops': attn_flops,
                'memory': 0
            })
            
            # LayerNorm + MLP
            layers.append({
                'layer_name': f'block{i}_ln2',
                'layer_type': 'LayerNorm',
                'output_shape': (seq_len, d_model),
                'params': 2 * d_model,
                'flops': 5 * seq_len * d_model,
                'memory': 0
            })
            
            ffn_hidden = d_model * 4
            ffn_params = d_model * ffn_hidden + ffn_hidden + ffn_hidden * d_model + d_model
            ffn_flops = 2 * seq_len * (d_model * ffn_hidden + ffn_hidden * d_model)
            
            layers.append({
                'layer_name': f'block{i}_mlp',
                'layer_type': 'MLP',
                'output_shape': (seq_len, d_model),
                'params': ffn_params,
                'flops': ffn_flops,
                'memory': 0
            })
        
        # Classification Head
        layers.append({
            'layer_name': 'head',
            'layer_type': 'Linear',
            'output_shape': (1000,),
            'params': d_model * 1000 + 1000,  # 769,000
            'flops': 2 * d_model * 1000,
            'memory': 0
        })
        
        return layers
    
    # 默认返回空
    return []


def _generate_network_report(network_name: str, input_size: int, batch_size: int, 
                             layers_data: List[Dict], total_params: int, total_flops: int) -> str:
    """生成网络分析报告"""
    
    report = f"""# {network_name} 网络分析报告

## 基本信息
- **网络名称**: {network_name}
- **输入尺寸**: [{batch_size}, 3, {input_size}, {input_size}]
- **总参数量**: {total_params:,} ({total_params/1e6:.2f}M)
- **总FLOPs**: {total_flops:,} ({total_flops/1e9:.2f}G)
- **参数内存**: {total_params*4/1024/1024:.2f} MB (FP32)

## 逐层详细信息

| 层名称 | 层类型 | 输出形状 | 参数量 | FLOPs |
|--------|--------|----------|--------|-------|
"""
    
    for layer in layers_data:
        params_str = f"{layer['params']/1e6:.2f}M" if layer['params'] > 1e6 else f"{layer['params']:,}"
        flops_str = f"{layer['flops']/1e9:.2f}G" if layer['flops'] > 1e9 else f"{layer['flops']/1e6:.2f}M"
        report += f"| {layer['layer_name']} | {layer['layer_type']} | {layer['output_shape']} | {params_str} | {flops_str} |\n"
    
    report += f"""
## 性能评估

### 参数量分析
- 总参数量较{'大' if total_params > 50e6 else '小'}，{'可能' if total_params > 50e6 else '不'}需要模型压缩
- 平均每层参数量: {total_params/len(layers_data):,.0f}

### 计算复杂度
- FLOPs: {total_flops/1e9:.2f}G
- 估计推理时间 (1080Ti): ~{total_flops/1e12*10:.2f}ms

### 内存占用
- 模型参数: {total_params*4/1024/1024:.2f} MB
- 估计峰值内存: ~{total_params*4/1024/1024*3:.2f} MB (包含梯度和优化器状态)

---
*报告生成时间: {pd.Timestamp.now()}*
"""
    
    return report
