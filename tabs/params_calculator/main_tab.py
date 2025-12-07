"""参数计算器主标签页

提供单层分析和网络分析的界面功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import streamlit as st

from .layer_analyzer import LayerAnalyzer
from .network_analysis import full_network_analysis


def params_calculator_tab():
    """参数量与FLOPs计算器标签页"""

    st.header("🔢 参数量与FLOPs计算器")

    st.markdown(
        """
    ### 核心功能：逐层分析网络计算细节
    
    输入网络层的配置，自动计算：
    - 📊 参数量（Params）
    - 📈 浮点运算量（FLOPs / MACs）
    - 💾 内存占用（前向/反向传播）
    - 🔍 输出特征图尺寸
    
    **与 torchinfo 的区别**：我们不仅给出数字，还展示**每个数字背后的计算公式**！
    """
    )

    # 选择分析模式
    analysis_mode = st.radio(
        "选择分析模式",
        ["单层分析", "完整网络分析"],
        horizontal=True,
        key="analysis_mode"
    )

    if analysis_mode == "单层分析":
        _single_layer_analysis()
    else:
        full_network_analysis()


def _single_layer_analysis():
    """单层分析模式"""
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
            "LayerNorm (层归一化)",
        ],
    )

    analyzer = LayerAnalyzer()

    if "Conv2d" in layer_type:
        _conv2d_analysis(analyzer)
    elif "DepthwiseConv2d" in layer_type:
        _depthwise_conv_analysis(analyzer)
    elif "Linear" in layer_type:
        _linear_analysis(analyzer)
    elif "MultiHeadAttention" in layer_type:
        _attention_analysis(analyzer)
    elif "LSTM" in layer_type:
        _lstm_analysis(analyzer)
    elif "Embedding" in layer_type:
        _embedding_analysis(analyzer)
    elif "BatchNorm2d" in layer_type:
        _batchnorm_analysis(analyzer)
    elif "LayerNorm" in layer_type:
        _layernorm_analysis(analyzer)


def _conv2d_analysis(analyzer):
    """Conv2d层分析"""
    st.markdown("### 🖼️ Conv2d 卷积层分析")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**输入配置**")
        C_in = st.number_input(
            "输入通道数 (in_channels)", min_value=1, value=3, step=1
        )
        H_in = st.number_input("输入高度 (H)", min_value=1, value=224, step=1)
        W_in = st.number_input("输入宽度 (W)", min_value=1, value=224, step=1)

    with col2:
        st.markdown("**层参数配置**")
        C_out = st.number_input(
            "输出通道数 (out_channels)", min_value=1, value=64, step=1
        )
        kernel_size = st.number_input(
            "卷积核大小 (kernel_size)", min_value=1, value=7, step=1
        )
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
        use_bias=use_bias,
    )

    # 显示结果
    _display_conv2d_results(result, C_in, H_in, W_in)


def _display_conv2d_results(result, C_in, H_in, W_in):
    """显示Conv2d分析结果"""
    st.markdown("---")
    st.markdown("### 📊 分析结果")

    # 输出形状
    st.markdown("#### 1️⃣ 输出特征图尺寸")
    C_out_calc, H_out, W_out = result["output_shape"]

    st.latex(
        r"H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1"
    )
    st.latex(
        r"W_{out} = \left\lfloor \frac{W_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1"
    )

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
    
    weight_params = result["parameters"]["weight"]
    bias_params = result["parameters"]["bias"]
    total_params = result["parameters"]["total"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("权重参数", f"{weight_params:,}")
    with col2:
        st.metric("偏置参数", f"{bias_params:,}")
    with col3:
        st.metric("总参数量", f"{total_params:,}")

    # FLOPs
    st.markdown("#### 3️⃣ FLOPs计算")
    st.latex(r"FLOPs = 2 \times MACs = 2 \times C_{out} \times H_{out} \times W_{out} \times K_h \times K_w \times C_{in}")
    
    total_flops = result["flops"]["total"]
    macs = result["flops"]["macs"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MACs", macs)
    with col2:
        st.metric("FLOPs", total_flops)

    # 内存占用
    st.markdown("#### 4️⃣ 内存占用")
    param_memory = result["memory_mb"]["parameters"]
    forward_memory = result["memory_mb"]["forward"]
    backward_memory = result["memory_mb"]["backward"]
    total_memory = result["memory_mb"]["total"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("参数内存", f"{param_memory:.2f}MB")
    with col2:
        st.metric("前向内存", f"{forward_memory:.2f}MB")
    with col3:
        st.metric("总内存", f"{total_memory:.2f}MB")


def _depthwise_conv_analysis(analyzer):
    """深度可分离卷积分析"""
    st.markdown("### 📱 DepthwiseConv2d 深度可分离卷积分析")
    
    # 配置界面
    col1, col2 = st.columns(2)

    with col1:
        in_channels = st.number_input("输入通道数", 1, 1024, 32)
        kernel_size = st.number_input("卷积核大小", 1, 11, 3)
        stride = st.number_input("步长", 1, 4, 1)
        padding = st.number_input("填充", 0, 5, 1)

    with col2:
        H_in = st.number_input("输入高度", 16, 512, 224)
        W_in = st.number_input("输入宽度", 16, 512, 224)
        use_bias = st.checkbox("使用偏置", True)

    # 计算分析
    result = analyzer.depthwise_conv2d_analysis(
        in_channels, kernel_size, stride, padding, (in_channels, H_in, W_in), use_bias
    )

    # 显示结果
    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("参数量", f"{result['parameters']['total']:,}")
        st.metric("内存占用", f"{result['memory_mb']['parameters']:.2f}MB")
    with col2:
        st.metric("FLOPs", result['flops']['flops_readable'])
        st.metric("输出形状", str(result['output_shape']))


def _linear_analysis(analyzer):
    """全连接层分析"""
    st.markdown("### 🔗 Linear 全连接层分析")
    
    in_features = st.number_input("输入特征数", 1, 4096, 512)
    out_features = st.number_input("输出特征数", 1, 4096, 512)
    use_bias = st.checkbox("使用偏置", True)

    result = analyzer.linear_analysis(in_features, out_features, use_bias)

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("权重参数", f"{result['parameters']['weight']:,}")
    with col2:
        st.metric("总参数量", f"{result['parameters']['total']:,}")
    with col3:
        st.metric("FLOPs", result['flops']['flops_readable'])


def _attention_analysis(analyzer):
    """多头注意力分析"""
    st.markdown("### 👁️ MultiHeadAttention 多头注意力分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        d_model = st.number_input("模型维度", 64, 2048, 512)
    with col2:
        num_heads = st.number_input("注意力头数", 1, 32, 8)
    with col3:
        seq_len = st.number_input("序列长度", 16, 1024, 128)

    has_qkv_bias = st.checkbox("QKV使用偏置", True)

    result = analyzer.attention_analysis(d_model, num_heads, seq_len, has_qkv_bias)

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总参数量", f"{result['parameters']['total']:,}")
    with col2:
        st.metric("FLOPs", result['flops']['flops_readable'])
    with col3:
        st.metric("注意力矩阵内存", f"{result['memory_mb']['attention_matrix']:.2f}MB")


def _lstm_analysis(analyzer):
    """LSTM分析"""
    st.markdown("### 🔄 LSTM 长短期记忆网络分析")
    
    col1, col2 = st.columns(2)
    with col1:
        input_size = st.number_input("输入维度", 64, 2048, 512)
        hidden_size = st.number_input("隐藏维度", 64, 2048, 512)
    with col2:
        num_layers = st.number_input("层数", 1, 8, 2)
        bidirectional = st.checkbox("双向", False)

    result = analyzer.lstm_analysis(input_size, hidden_size, num_layers, bidirectional=True)

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总参数量", f"{result['parameters']['total']:,}")
    with col2:
        st.metric("每层参数", f"{result['parameters']['per_layer']:,}")
    with col3:
        st.metric("每时间步FLOPs", result['flops']['flops_readable'])


def _embedding_analysis(analyzer):
    """嵌入层分析"""
    st.markdown("### 📚 Embedding 嵌入层分析")
    
    col1, col2 = st.columns(2)
    with col1:
        num_embeddings = st.number_input("词表大小", 1000, 100000, 10000)
    with col2:
        embedding_dim = st.number_input("嵌入维度", 64, 1024, 512)

    result = analyzer.embedding_analysis(num_embeddings, embedding_dim)

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("参数量", f"{result['parameters']['total']:,}")
    with col2:
        st.metric("内存占用", f"{result['memory_mb']['parameters']:.2f}MB")


def _batchnorm_analysis(analyzer):
    """批归一化分析"""
    st.markdown("### 📊 BatchNorm2d 批归一化分析")
    
    col1, col2 = st.columns(2)
    with col1:
        num_features = st.number_input("特征数", 16, 1024, 64)
    with col2:
        H = st.number_input("高度", 16, 512, 224)
        W = st.number_input("宽度", 16, 512, 224)

    result = analyzer.batchnorm2d_analysis(num_features, (num_features, H, W))

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("参数量", f"{result['parameters']['total']:,}")
    with col2:
        st.metric("FLOPs", result['flops']['flops_readable'])


def _layernorm_analysis(analyzer):
    """层归一化分析"""
    st.markdown("### 📏 LayerNorm 层归一化分析")
    
    normalized_shape = st.number_input("归一化维度", 64, 2048, 512)
    
    # 假设输入形状
    input_shape = (normalized_shape, 128)  # (d_model, seq_len)

    result = analyzer.layernorm_analysis(normalized_shape, input_shape)

    st.markdown("---")
    st.markdown("### 📊 分析结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("参数量", f"{result['parameters']['total']:,}")
    with col2:
        st.metric("FLOPs", result['flops']['flops_readable'])