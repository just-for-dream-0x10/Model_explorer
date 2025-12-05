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
        ["Conv2d (卷积层)", "Linear (全连接层)", "BatchNorm2d (批归一化)"]
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
