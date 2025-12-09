"""
内存分析标签页
Memory Analysis Tab

分析神经网络训练时的内存占用
核心理念：让你看到每一层到底占用多少内存
"""

import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from utils.memory_analyzer import (
    analyze_conv2d_memory,
    analyze_linear_memory,
    analyze_batchnorm_memory,
    analyze_pooling_memory,
    analyze_model_memory,
    LayerMemoryInfo,
)


def plot_memory_breakdown(layers_info):
    """
    绘制内存占用分解饼图

    Args:
        layers_info: 层级内存信息列表

    Returns:
        fig: Plotly图表
    """
    # 按层类型汇总
    type_memory = {}
    for layer in layers_info:
        if layer.layer_type not in type_memory:
            type_memory[layer.layer_type] = 0
        type_memory[layer.layer_type] += layer.backward_peak

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(type_memory.keys()),
                values=list(type_memory.values()),
                hole=0.3,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(title="内存占用分解（按层类型）", height=400)

    return fig


def plot_layer_memory_bars(layers_info, top_n=10):
    """
    绘制各层内存占用柱状图

    Args:
        layers_info: 层级内存信息列表
        top_n: 显示前N个内存占用最大的层

    Returns:
        fig: Plotly图表
    """
    # 按反向峰值内存排序
    sorted_layers = sorted(layers_info, key=lambda x: x.backward_peak, reverse=True)[
        :top_n
    ]

    layer_names = [f"{layer.name}\n({layer.layer_type})" for layer in sorted_layers]
    forward_mem = [layer.forward_peak for layer in sorted_layers]
    backward_mem = [layer.backward_peak for layer in sorted_layers]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=layer_names, y=forward_mem, name="前向峰值", marker_color="lightblue")
    )

    fig.add_trace(
        go.Bar(x=layer_names, y=backward_mem, name="反向峰值", marker_color="darkblue")
    )

    fig.update_layout(
        title=f"内存占用Top {top_n}层",
        xaxis_title="层",
        yaxis_title="内存 (MB)",
        barmode="group",
        height=500,
    )

    return fig


def plot_memory_composition(layer_info):
    """
    绘制单层内存组成

    Args:
        layer_info: 单层内存信息

    Returns:
        fig: Plotly图表
    """
    categories = ["输入激活值", "输出激活值", "参数", "梯度"]
    values = [
        layer_info.input_memory,
        layer_info.output_memory,
        layer_info.param_memory,
        layer_info.grad_memory,
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                text=[f"{v:.2f} MB" for v in values],
                textposition="auto",
                marker_color=["lightgreen", "green", "orange", "red"],
            )
        ]
    )

    fig.update_layout(
        title=f"{layer_info.name} 内存组成",
        xaxis_title="类型",
        yaxis_title="内存 (MB)",
        height=400,
    )

    return fig


def plot_cumulative_memory(layers_info):
    """
    绘制累计内存曲线

    Args:
        layers_info: 层级内存信息列表

    Returns:
        fig: Plotly图表
    """
    layer_indices = list(range(len(layers_info)))

    cumulative_forward = []
    cumulative_backward = []
    cumulative_param = []

    cum_f = 0
    cum_b = 0
    cum_p = 0

    for layer in layers_info:
        cum_f += layer.output_memory
        cum_b += layer.grad_memory
        cum_p += layer.param_memory

        cumulative_forward.append(cum_f)
        cumulative_backward.append(cum_b)
        cumulative_param.append(cum_p)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layer_indices,
            y=cumulative_forward,
            mode="lines+markers",
            name="累计前向激活值",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=layer_indices,
            y=cumulative_backward,
            mode="lines+markers",
            name="累计反向梯度",
            line=dict(color="red", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=layer_indices,
            y=cumulative_param,
            mode="lines+markers",
            name="累计参数",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title="累计内存占用曲线",
        xaxis_title="层索引",
        yaxis_title="累计内存 (MB)",
        height=500,
        hovermode="x unified",
    )

    return fig


def explain_memory_concepts():
    """解释内存相关概念"""
    st.markdown(
        """
    ### 💾 神经网络内存占用详解
    
    训练神经网络时，内存占用主要来自四个部分：
    
    #### 1. 前向激活值（Forward Activations）
    ```
    每一层的输出张量
    示例：Conv2d(3, 64, 7, stride=2)
    输入: [1, 3, 224, 224] → 输出: [1, 64, 112, 112]
    输出内存 = 1 × 64 × 112 × 112 × 4字节 = 3.06 MB
    ```
    
    **为什么需要保存**：反向传播时计算梯度需要用到
    
    #### 2. 反向梯度（Backward Gradients）
    ```
    每一层输出的梯度
    梯度形状 = 输出形状
    梯度内存 = 输出内存
    ```
    
    **为什么需要保存**：用于更新权重参数
    
    #### 3. 参数（Parameters）
    ```
    权重和偏置
    示例：Linear(512, 1000)
    权重: [1000, 512] + 偏置: [1000]
    参数内存 = (512000 + 1000) × 4字节 = 1.96 MB
    ```
    
    **训练和推理都需要**
    
    #### 4. 参数梯度（Parameter Gradients）
    ```
    参数的梯度
    形状与参数相同
    参数梯度内存 = 参数内存
    ```
    
    #### 内存计算公式
    
    **前向传播内存**：
    ```
    前向内存 = Σ(每层的输出激活值) + 参数内存
    ```
    
    **反向传播内存**：
    ```
    反向内存 = 前向内存 + Σ(每层的梯度) + 参数梯度
    ```
    
    **峰值内存**：
    ```
    峰值内存 = max(各层的反向峰值)
    通常是最大的层
    ```
    
    #### 内存优化技巧
    
    1. **梯度检查点（Gradient Checkpointing）**
       - 不保存所有中间激活值
       - 需要时重新计算
       - 用时间换内存
    
    2. **混合精度训练（Mixed Precision）**
       - 使用float16代替float32
       - 内存减半
       - 需要注意数值稳定性
    
    3. **降低Batch Size**
       - 最直接的方法
       - 但可能影响训练效果
    
    4. **梯度累积（Gradient Accumulation）**
       - 小batch多次前向，累积梯度
       - 模拟大batch效果
    """
    )


def memory_analysis_tab(chinese_supported=True):
    """内存分析主函数"""

    st.header("💾 内存分析器")
    st.markdown(
        """
    > **核心功能**：分析神经网络训练时的内存占用，定位内存瓶颈
    
    **分析维度**：前向激活值、反向梯度、参数、峰值内存
    """
    )

    st.markdown("---")

    # 内存概念解释
    with st.expander("💡 内存占用详解（点击展开）", expanded=False):
        explain_memory_concepts()

    st.markdown("---")

    # 分析模式选择
    st.subheader("🔧 选择分析模式")

    analysis_mode = st.radio("分析模式", ["单层分析", "模型分析"], horizontal=True)

    if analysis_mode == "单层分析":
        st.markdown("---")
        st.subheader("📐 单层内存分析")

        layer_type = st.selectbox(
            "选择层类型", ["Conv2d", "Linear", "BatchNorm2d", "MaxPool2d"]
        )

        if layer_type == "Conv2d":
            col1, col2 = st.columns(2)

            with col1:
                in_channels = st.number_input("输入通道数", 1, 1024, 3)
                out_channels = st.number_input("输出通道数", 1, 1024, 64)
                kernel_size = st.number_input("卷积核大小", 1, 11, 3)

            with col2:
                batch_size = st.number_input("Batch Size", 1, 128, 1)
                input_h = st.number_input("输入高度", 1, 512, 224)
                input_w = st.number_input("输入宽度", 1, 512, 224)
                stride = st.number_input("步长", 1, 4, 1)
                padding = st.number_input("填充", 0, 10, 1)

            if st.button("🔍 分析Conv2d内存"):
                with st.spinner("计算中..."):
                    info = analyze_conv2d_memory(
                        in_channels,
                        out_channels,
                        (kernel_size, kernel_size),
                        (batch_size, in_channels, input_h, input_w),
                        stride,
                        padding,
                    )

                st.success("✅ 分析完成！")

                # 显示基本信息
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**输入输出形状**")
                    st.code(
                        f"""
输入: {info.input_shape}
输出: {info.output_shape}
                    """
                    )

                with col2:
                    st.markdown("**参数信息**")
                    st.code(
                        f"""
参数数量: {info.param_count:,}
参数内存: {info.param_memory:.2f} MB
                    """
                    )

                # 内存统计
                st.markdown("---")
                st.markdown("#### 💾 内存占用详情")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("输入激活值", f"{info.input_memory:.2f} MB")
                with col2:
                    st.metric("输出激活值", f"{info.output_memory:.2f} MB")
                with col3:
                    st.metric("梯度内存", f"{info.grad_memory:.2f} MB")
                with col4:
                    st.metric(
                        "峰值内存", f"{info.backward_peak:.2f} MB", delta="反向传播"
                    )

                # 可视化
                fig = plot_memory_composition(info)
                st.plotly_chart(fig, use_container_width=True)

                # 详细分解
                st.markdown("#### 📊 内存计算分解")
                st.markdown(
                    f"""
                **前向传播**：
                - 输入激活值：`{info.input_shape}` × 4 bytes = `{info.input_memory:.2f} MB`
                - 输出激活值：`{info.output_shape}` × 4 bytes = `{info.output_memory:.2f} MB`
                - 参数：`[{out_channels}, {in_channels}, {kernel_size}, {kernel_size}]` × 4 bytes = `{info.param_memory:.2f} MB`
                - **前向峰值**：`{info.forward_peak:.2f} MB`
                
                **反向传播**：
                - 前向内存：`{info.forward_peak:.2f} MB`
                - 梯度内存：`{info.grad_memory:.2f} MB`
                - **反向峰值**：`{info.backward_peak:.2f} MB`
                """
                )

        elif layer_type == "Linear":
            col1, col2 = st.columns(2)

            with col1:
                in_features = st.number_input("输入特征数", 1, 10000, 512)
                out_features = st.number_input("输出特征数", 1, 10000, 1000)

            with col2:
                batch_size = st.number_input("Batch Size", 1, 128, 1)

            if st.button("🔍 分析Linear内存"):
                with st.spinner("计算中..."):
                    info = analyze_linear_memory(
                        in_features, out_features, (batch_size, in_features)
                    )

                st.success("✅ 分析完成！")

                # 显示结果（与Conv2d类似）
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("输入内存", f"{info.input_memory:.4f} MB")
                with col2:
                    st.metric("输出内存", f"{info.output_memory:.4f} MB")
                with col3:
                    st.metric("参数内存", f"{info.param_memory:.2f} MB")
                with col4:
                    st.metric("峰值内存", f"{info.backward_peak:.2f} MB")

                fig = plot_memory_composition(info)
                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    f"""
                **参数量**: {info.param_count:,}个
                
                **计算**：权重 `[{out_features}, {in_features}]` + 偏置 `[{out_features}]`
                = `{in_features * out_features + out_features:,}` 个参数
                """
                )

    else:  # 模型分析
        st.markdown("---")
        st.subheader("🏗️ 完整模型内存分析")

        st.info("💡 提示：选择预定义模型或自定义简单模型进行分析")

        model_choice = st.selectbox(
            "选择模型",
            [
                "ResNet-18 (简化)",
                "MobileNet-V2 (简化)",
                "ViT-Tiny (简化)",
                "自定义模型",
            ],
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", 1, 64, 1, key="model_batch")
        with col2:
            input_size = st.number_input("输入尺寸", 32, 512, 224, key="model_input")

        if st.button("🚀 开始分析", type="primary"):
            with st.spinner("分析中...这可能需要几秒钟"):
                # 创建简化模型
                if "ResNet" in model_choice:
                    model = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(256, 1000),
                    )
                elif "MobileNet" in model_choice:
                    model = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 64, 3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(128, 1000),
                    )
                else:  # ViT
                    model = nn.Sequential(
                        nn.Conv2d(3, 192, 16, stride=16),  # Patch Embedding
                        nn.Flatten(2),
                        nn.Linear(192, 192),
                        nn.Linear(192, 1000),
                    )

                result = analyze_model_memory(
                    model, (batch_size, 3, input_size, input_size), detailed=True
                )

            st.success("✅ 分析完成！")

            # 显示总结
            st.markdown("#### 📊 内存总结")

            summary = result["summary"]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("前向内存", f"{summary['total_forward_memory']:.2f} MB")
            with col2:
                st.metric("反向内存", f"{summary['total_backward_memory']:.2f} MB")
            with col3:
                st.metric("参数内存", f"{summary['total_param_memory']:.2f} MB")
            with col4:
                st.metric(
                    "峰值内存", f"{summary['peak_memory']:.2f} MB", delta="训练时"
                )

            # 瓶颈层
            if result["bottleneck"]["layer"]:
                st.warning(
                    f"""
                ⚠️ **内存瓶颈层**: `{result['bottleneck']['layer'].name}`
                - 占比: **{result['bottleneck']['percentage']:.1f}%**
                - 峰值内存: **{result['bottleneck']['layer'].backward_peak:.2f} MB**
                """
                )

            # 可视化
            if result["layers"]:
                st.markdown("---")
                st.markdown("#### 📈 内存可视化")

                tab1, tab2, tab3 = st.tabs(["层级对比", "类型分解", "累计曲线"])

                with tab1:
                    fig1 = plot_layer_memory_bars(
                        result["layers"], top_n=min(10, len(result["layers"]))
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with tab2:
                    fig2 = plot_memory_breakdown(result["layers"])
                    st.plotly_chart(fig2, use_container_width=True)

                with tab3:
                    fig3 = plot_cumulative_memory(result["layers"])
                    st.plotly_chart(fig3, use_container_width=True)

                # 详细逐层分析
                st.markdown("---")
                st.markdown("#### 🔍 逐层内存详细分析")

                # 选择一个模型查看详情
                selected_model_for_detail = st.selectbox(
                    "选择模型查看逐层详情",
                    list(results.keys()),
                    key="detail_model_select",
                )

                if results[selected_model_for_detail]["layers"]:
                    layers = results[selected_model_for_detail]["layers"]

                    st.markdown(f"**{selected_model_for_detail} 逐层内存分析**")

                    # 创建详细表格
                    table_data = {
                        "层索引": [],
                        "层名称": [],
                        "类型": [],
                        "输入形状": [],
                        "输出形状": [],
                        "参数数量": [],
                        "参数内存(MB)": [],
                        "激活值内存(MB)": [],
                        "梯度内存(MB)": [],
                        "峰值内存(MB)": [],
                    }

                    for idx, layer in enumerate(layers):
                        table_data["层索引"].append(idx)
                        table_data["层名称"].append(layer.name)
                        table_data["类型"].append(layer.layer_type)
                        table_data["输入形状"].append(
                            str(layer.input_shape) if layer.input_shape else "N/A"
                        )
                        table_data["输出形状"].append(
                            str(layer.output_shape) if layer.output_shape else "N/A"
                        )
                        table_data["参数数量"].append(f"{layer.param_count:,}")
                        table_data["参数内存(MB)"].append(f"{layer.param_memory:.4f}")
                        table_data["激活值内存(MB)"].append(
                            f"{layer.output_memory:.4f}"
                        )
                        table_data["梯度内存(MB)"].append(f"{layer.grad_memory:.4f}")
                        table_data["峰值内存(MB)"].append(f"{layer.backward_peak:.4f}")

                    st.dataframe(table_data, use_container_width=True)

                    # 选择某一层查看详细计算
                    st.markdown("---")
                    st.markdown("#### 🔬 单层内存计算详解")

                    layer_idx = st.selectbox(
                        "选择层查看详细计算过程",
                        range(len(layers)),
                        format_func=lambda x: f"Layer {x}: {layers[x].name} ({layers[x].layer_type})",
                        key="layer_detail_select",
                    )

                    selected_layer = layers[layer_idx]

                    st.markdown(f"**Layer {layer_idx}: {selected_layer.name}**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**形状信息**")
                        st.code(
                            f"""
类型: {selected_layer.layer_type}
输入形状: {selected_layer.input_shape}
输出形状: {selected_layer.output_shape}
参数数量: {selected_layer.param_count:,}
                        """
                        )

                    with col2:
                        st.markdown("**内存占用**")
                        st.code(
                            f"""
输入内存: {selected_layer.input_memory:.4f} MB
输出内存: {selected_layer.output_memory:.4f} MB
参数内存: {selected_layer.param_memory:.4f} MB
梯度内存: {selected_layer.grad_memory:.4f} MB
前向峰值: {selected_layer.forward_peak:.4f} MB
反向峰值: {selected_layer.backward_peak:.4f} MB
                        """
                        )

                    # 详细计算过程
                    st.markdown("**详细计算过程**")

                    if (
                        selected_layer.layer_type == "Conv2d"
                        and selected_layer.input_shape
                    ):
                        B, C_in, H_in, W_in = selected_layer.input_shape
                        B_out, C_out, H_out, W_out = selected_layer.output_shape

                        st.markdown(
                            f"""
                        **Conv2d层内存计算**：
                        
                        1. **输入内存**：
                        ```
                        形状: [{B}, {C_in}, {H_in}, {W_in}]
                        元素数: {B} × {C_in} × {H_in} × {W_in} = {B*C_in*H_in*W_in:,}
                        内存: {B*C_in*H_in*W_in:,} × 4字节 / 1024² = {selected_layer.input_memory:.4f} MB
                        ```
                        
                        2. **输出内存（激活值）**：
                        ```
                        形状: [{B_out}, {C_out}, {H_out}, {W_out}]
                        元素数: {B_out} × {C_out} × {H_out} × {W_out} = {B_out*C_out*H_out*W_out:,}
                        内存: {B_out*C_out*H_out*W_out:,} × 4字节 / 1024² = {selected_layer.output_memory:.4f} MB
                        ```
                        
                        3. **参数内存**：
                        ```
                        参数数量: {selected_layer.param_count:,}
                        内存: {selected_layer.param_count:,} × 4字节 / 1024² = {selected_layer.param_memory:.4f} MB
                        ```
                        
                        4. **梯度内存**：
                        ```
                        梯度形状 = 输出形状 = [{B_out}, {C_out}, {H_out}, {W_out}]
                        梯度内存 = 输出内存 = {selected_layer.grad_memory:.4f} MB
                        ```
                        
                        5. **峰值内存（反向传播时）**：
                        ```
                        峰值 = 输入内存 + 输出内存 + 参数内存 + 梯度内存
                             = {selected_layer.input_memory:.4f} + {selected_layer.output_memory:.4f} + {selected_layer.param_memory:.4f} + {selected_layer.grad_memory:.4f}
                             = {selected_layer.backward_peak:.4f} MB
                        ```
                        """
                        )

                    elif (
                        selected_layer.layer_type == "Linear"
                        and selected_layer.input_shape
                    ):
                        B, in_features = selected_layer.input_shape
                        B_out, out_features = selected_layer.output_shape

                        st.markdown(
                            f"""
                        **Linear层内存计算**：
                        
                        1. **输入内存**：
                        ```
                        形状: [{B}, {in_features}]
                        元素数: {B} × {in_features} = {B*in_features:,}
                        内存: {B*in_features:,} × 4字节 / 1024² = {selected_layer.input_memory:.4f} MB
                        ```
                        
                        2. **输出内存**：
                        ```
                        形状: [{B_out}, {out_features}]
                        元素数: {B_out} × {out_features} = {B_out*out_features:,}
                        内存: {B_out*out_features:,} × 4字节 / 1024² = {selected_layer.output_memory:.4f} MB
                        ```
                        
                        3. **参数内存**：
                        ```
                        权重: [{out_features}, {in_features}] = {out_features*in_features:,}
                        偏置: [{out_features}] = {out_features:,}
                        总参数: {selected_layer.param_count:,}
                        内存: {selected_layer.param_count:,} × 4字节 / 1024² = {selected_layer.param_memory:.4f} MB
                        ```
                        
                        4. **峰值内存**：
                        ```
                        峰值 = {selected_layer.input_memory:.4f} + {selected_layer.output_memory:.4f} + {selected_layer.param_memory:.4f} + {selected_layer.grad_memory:.4f}
                             = {selected_layer.backward_peak:.4f} MB
                        ```
                        """
                        )

                    else:
                        st.info(
                            f"层类型: {selected_layer.layer_type}，内存占用已计算，详细公式请参考上方的计算依据说明。"
                        )

    # 多模型对比功能
    st.markdown("---")
    st.subheader("🔬 多模型内存对比")

    st.markdown(
        """
    **对比不同架构的内存使用差异**：
    - CNN vs Transformer的内存特点
    - 自定义模型配置（层数、token数等）
    - 参数量与内存占用的关系
    """
    )

    # 配置面板
    st.markdown("#### ⚙️ 模型配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**CNN配置**")
        cnn_layers = st.slider("CNN层数", 3, 20, 5, key="cnn_layers")
        cnn_channels = st.slider("通道数", 32, 256, 64, key="cnn_channels")

    with col2:
        st.markdown("**Transformer配置**")
        num_layers = st.slider("Transformer层数", 1, 24, 6, key="tf_layers")
        seq_length = st.slider("序列长度(tokens)", 16, 512, 100, key="seq_len")
        hidden_dim = st.slider("隐藏维度", 192, 1024, 384, key="hidden_dim")

    with col3:
        st.markdown("**通用配置**")
        batch_size_compare = st.slider("Batch Size", 1, 32, 4, key="batch_compare")
        input_size_compare = st.selectbox(
            "图像尺寸", [112, 224, 384], index=1, key="img_size"
        )

    if st.button("🚀 开始多模型对比", type="primary", key="multi_model"):
        with st.spinner("分析中..."):
            # 构建自定义模型
            models_config = {}

            # CNN模型（基于配置）
            cnn_layers_list = []
            current_channels = 3
            for i in range(cnn_layers):
                out_channels = min(cnn_channels * (2 ** (i // 2)), 512)
                stride = 2 if i % 2 == 0 and i < cnn_layers - 1 else 1
                cnn_layers_list.append(
                    nn.Conv2d(
                        current_channels, out_channels, 3, stride=stride, padding=1
                    )
                )
                cnn_layers_list.append(nn.BatchNorm2d(out_channels))
                cnn_layers_list.append(nn.ReLU())
                current_channels = out_channels

            cnn_layers_list.extend(
                [
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(current_channels, 1000),
                ]
            )

            models_config[f"CNN-{cnn_layers}层"] = nn.Sequential(*cnn_layers_list)

            # Transformer模型（基于配置）
            # 简化版：只分析Patch Embedding部分，后续层用Linear模拟
            # 使用动态参数建议器
    from utils.parameter_suggester import get_suggested_params

    try:
        # 获取用户选择的图像尺寸
        img_size = input_size_compare
        # 根据图像大小动态建议patch size
        if img_size <= 64:
            patch_size = 8
        elif img_size <= 128:
            patch_size = 16
        else:
            patch_size = 32
    except Exception as e:
        # 如果动态计算失败，使用默认值
        patch_size = 16

        tf_layers_list = [
            nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size),  # Patch Embedding
            nn.Flatten(2),
        ]

        # 添加Transformer层（简化为Linear模拟）
        for i in range(num_layers):
            tf_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            tf_layers_list.append(nn.LayerNorm(hidden_dim))

        models_config[f"Transformer-{num_layers}层"] = nn.Sequential(*tf_layers_list)

        # 轻量级模型
        models_config["轻量级CNN"] = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1000),
        )

        # 分析所有模型
        results = {}
        for model_name, model in models_config.items():
            result = analyze_model_memory(
                model,
                (batch_size_compare, 3, input_size_compare, input_size_compare),
                detailed=True,
            )
            results[model_name] = result

        st.success("✅ 对比完成！")

        # 首先展示模型结构
        st.markdown("#### 🏗️ 模型架构详情")

        with st.expander("📋 查看各模型的详细结构", expanded=False):
            for model_name, model in models_config.items():
                st.markdown(f"**{model_name}**")
                st.code(str(model))

                # 显示参数量
                param_count = sum(p.numel() for p in model.parameters())
                st.info(f"总参数量: {param_count:,} ({param_count/1e6:.2f}M)")
                st.markdown("---")

        # 计算依据说明
        st.markdown("#### 📐 内存计算依据")

        with st.expander("🔍 点击查看详细计算公式", expanded=True):
            st.markdown(
                """
            ### 内存计算公式
            
            #### 1. 张量内存计算
            ```python
            内存(MB) = 元素数量 × 每元素字节数 / (1024²)
            
            # 例如：float32张量 [1, 64, 112, 112]
            内存 = 1 × 64 × 112 × 112 × 4字节 / 1024² 
                 = 3,211,264 × 4 / 1,048,576
                 = 12.25 MB
            ```
            
            #### 2. Conv2d层内存
            ```python
            # 输入: [B, C_in, H, W]
            # 输出: [B, C_out, H_out, W_out]
            
            H_out = (H + 2×padding - kernel_size) // stride + 1
            W_out = (W + 2×padding - kernel_size) // stride + 1
            
            输入内存 = B × C_in × H × W × 4字节
            输出内存 = B × C_out × H_out × W_out × 4字节
            参数内存 = (C_out × C_in × K × K + C_out) × 4字节
            梯度内存 = 输出内存（反向传播时）
            
            峰值内存 = 输入内存 + 输出内存 + 参数内存 + 梯度内存
            ```
            
            #### 3. Linear层内存
            ```python
            # 输入: [B, in_features]
            # 输出: [B, out_features]
            
            输入内存 = B × in_features × 4字节
            输出内存 = B × out_features × 4字节
            参数内存 = (out_features × in_features + out_features) × 4字节
            梯度内存 = 输出内存
            ```
            
            #### 4. BatchNorm层内存
            ```python
            # 输入输出形状相同
            
            输入内存 = 输出内存
            参数内存 = num_features × 4 × 4字节  # gamma, beta, mean, var
            梯度内存 = 输出内存
            ```
            
            #### 5. 总内存计算
            ```python
            前向激活值内存 = Σ(每层的输出内存)
            反向梯度内存 = Σ(每层的梯度内存)
            参数内存 = Σ(每层的参数内存)
            
            训练峰值内存 = max(每层的峰值内存)
            推理内存 = 前向激活值内存 + 参数内存
            ```
            
            #### 6. 数据类型影响
            ```python
            float32: 4字节/元素（默认）
            float16: 2字节/元素（混合精度）
            int8:    1字节/元素（量化）
            
            # 使用float16可以减半内存占用
            ```
            
            ### 实际例子
            
            **Conv2d(3, 64, 7, stride=2) with input [1, 3, 224, 224]**
            ```
            输出形状: [1, 64, 112, 112]
            
            输入内存 = 1×3×224×224×4 / 1024² = 0.57 MB
            输出内存 = 1×64×112×112×4 / 1024² = 3.06 MB
            参数内存 = (64×3×7×7 + 64)×4 / 1024² = 0.04 MB
            梯度内存 = 3.06 MB
            
            峰值内存 = 0.57 + 3.06 + 0.04 + 3.06 = 6.73 MB
            ```
            
            ### 注意事项
            
            1. **本工具假设**：
               - 数据类型：float32 (4字节)
               - 保存所有中间激活值（用于反向传播）
               - 不考虑梯度检查点等优化
            
            2. **实际可能更大**：
               - 框架开销（PyTorch/TensorFlow）
               - 临时缓冲区
               - 优化器状态（Adam需要2×参数内存）
            
            3. **实际可能更小**：
               - 梯度检查点（重计算激活值）
               - 混合精度训练（FP16）
               - 内存优化技巧
            """
            )

        # 显示对比表格
        st.markdown("#### 📊 模型内存对比总览")

        comparison_data = {
            "模型": [],
            "前向内存(MB)": [],
            "反向内存(MB)": [],
            "参数内存(MB)": [],
            "峰值内存(MB)": [],
            "层数": [],
        }

        for model_name, result in results.items():
            summary = result["summary"]
            comparison_data["模型"].append(model_name)
            comparison_data["前向内存(MB)"].append(
                f"{summary['total_forward_memory']:.2f}"
            )
            comparison_data["反向内存(MB)"].append(
                f"{summary['total_backward_memory']:.2f}"
            )
            comparison_data["参数内存(MB)"].append(
                f"{summary['total_param_memory']:.2f}"
            )
            comparison_data["峰值内存(MB)"].append(f"{summary['peak_memory']:.2f}")
            comparison_data["层数"].append(summary["num_layers"])

        st.dataframe(comparison_data, use_container_width=True)

        # 可视化对比
        st.markdown("#### 📈 内存对比可视化")

        tab1, tab2, tab3 = st.tabs(["堆叠柱状图", "分组柱状图", "饼图对比"])

        models = list(results.keys())
        forward_mem = [results[m]["summary"]["total_forward_memory"] for m in models]
        backward_mem = [results[m]["summary"]["total_backward_memory"] for m in models]
        param_mem = [results[m]["summary"]["total_param_memory"] for m in models]
        peak_mem = [results[m]["summary"]["peak_memory"] for m in models]

        with tab1:
            # 堆叠柱状图 - 显示内存组成
            fig_stack = go.Figure()

            fig_stack.add_trace(
                go.Bar(
                    x=models,
                    y=param_mem,
                    name="参数内存",
                    marker_color="#FFA500",
                    text=[f"{v:.1f}MB" for v in param_mem],
                    textposition="inside",
                )
            )

            fig_stack.add_trace(
                go.Bar(
                    x=models,
                    y=forward_mem,
                    name="前向激活值",
                    marker_color="#87CEEB",
                    text=[f"{v:.1f}MB" for v in forward_mem],
                    textposition="inside",
                )
            )

            fig_stack.add_trace(
                go.Bar(
                    x=models,
                    y=backward_mem,
                    name="反向梯度",
                    marker_color="#4169E1",
                    text=[f"{v:.1f}MB" for v in backward_mem],
                    textposition="inside",
                )
            )

            fig_stack.update_layout(
                title=f"内存组成堆叠图 (Batch={batch_size_compare}, 输入={input_size_compare}×{input_size_compare})",
                xaxis_title="模型",
                yaxis_title="内存 (MB)",
                barmode="stack",
                height=500,
                showlegend=True,
            )

            st.plotly_chart(fig_stack, use_container_width=True)

            st.info(
                """
            **堆叠图解读**：
            - 🟧 橙色 = 参数内存（权重+偏置）
            - 🔵 浅蓝 = 前向激活值（中间结果）
            - 🔷 深蓝 = 反向梯度（训练时需要）
            - 柱子总高度 = 训练时总内存占用
            """
            )

        with tab2:
            # 分组柱状图 - 对比不同类型
            fig_group = go.Figure()

            fig_group.add_trace(
                go.Bar(
                    x=models, y=forward_mem, name="前向内存", marker_color="lightblue"
                )
            )

            fig_group.add_trace(
                go.Bar(x=models, y=backward_mem, name="反向内存", marker_color="blue")
            )

            fig_group.add_trace(
                go.Bar(x=models, y=param_mem, name="参数内存", marker_color="orange")
            )

            fig_group.add_trace(
                go.Bar(
                    x=models,
                    y=peak_mem,
                    name="峰值内存",
                    marker_color="red",
                    marker=dict(pattern=dict(shape="/")),
                )
            )

            fig_group.update_layout(
                title=f"内存类型对比 (Batch={batch_size_compare})",
                xaxis_title="模型",
                yaxis_title="内存 (MB)",
                barmode="group",
                height=500,
            )

            st.plotly_chart(fig_group, use_container_width=True)

        with tab3:
            # 饼图 - 每个模型的内存分解
            num_models = len(models)
            rows = (num_models + 1) // 2

            fig_pie = make_subplots(
                rows=rows,
                cols=2,
                subplot_titles=models,
                specs=[[{"type": "domain"}, {"type": "domain"}] for _ in range(rows)],
            )

            for idx, model_name in enumerate(models):
                row = idx // 2 + 1
                col = idx % 2 + 1

                fig_pie.add_trace(
                    go.Pie(
                        labels=["参数", "前向激活值", "反向梯度"],
                        values=[param_mem[idx], forward_mem[idx], backward_mem[idx]],
                        name=model_name,
                        marker=dict(colors=["#FFA500", "#87CEEB", "#4169E1"]),
                    ),
                    row=row,
                    col=col,
                )

            fig_pie.update_layout(title="各模型内存分解占比", height=300 * rows)

            st.plotly_chart(fig_pie, use_container_width=True)

            st.success(
                """
            **饼图解读**：
            - 一眼看出每个模型的内存"瓶颈"在哪里
            - CNN：通常前向激活值占比大
            - Transformer：参数内存占比相对更大
            """
            )

        # 参数量与内存的关系分析
        st.markdown("---")
        st.markdown("#### 🔍 参数量 vs 内存占用分析")

        # 计算参数量
        param_counts = {}
        for model_name, model in models_config.items():
            param_count = sum(p.numel() for p in model.parameters())
            param_counts[model_name] = param_count / 1e6  # 转为百万

        # 创建散点图
        fig_scatter = go.Figure()

        fig_scatter.add_trace(
            go.Scatter(
                x=[param_counts[m] for m in models],
                y=peak_mem,
                mode="markers+text",
                text=models,
                textposition="top center",
                marker=dict(
                    size=15, color=peak_mem, colorscale="Viridis", showscale=True
                ),
                name="模型",
            )
        )

        fig_scatter.update_layout(
            title="参数量 vs 峰值内存",
            xaxis_title="参数量 (Million)",
            yaxis_title="峰值内存 (MB)",
            height=500,
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # 关键发现
        st.markdown("#### 💡 关键发现")

        # 找出内存效率最高和最低的模型
        memory_efficiency = {}
        for model_name in models:
            if param_counts[model_name] > 0:
                efficiency = (
                    results[model_name]["summary"]["peak_memory"]
                    / param_counts[model_name]
                )
                memory_efficiency[model_name] = efficiency

        most_efficient = min(memory_efficiency, key=memory_efficiency.get)
        least_efficient = max(memory_efficiency, key=memory_efficiency.get)

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"""
            ✅ **内存效率最高**: {most_efficient}
            
            - 参数量: {param_counts[most_efficient]:.2f}M
            - 峰值内存: {results[most_efficient]['summary']['peak_memory']:.2f} MB
            - 效率比: {memory_efficiency[most_efficient]:.2f} MB/M参数
            """
            )

        with col2:
            st.warning(
                f"""
            ⚠️ **内存占用最大**: {least_efficient}
            
            - 参数量: {param_counts[least_efficient]:.2f}M
            - 峰值内存: {results[least_efficient]['summary']['peak_memory']:.2f} MB
            - 效率比: {memory_efficiency[least_efficient]:.2f} MB/M参数
            """
            )

        # 深入分析
        st.markdown("---")
        st.markdown("#### 📚 架构差异分析")

        st.markdown(
            """
        **CNN (ResNet/MobileNet) 特点**：
        - ✅ 参数内存占比相对较小
        - ✅ 激活值内存随空间分辨率变化
        - ✅ 早期层（大分辨率）内存占用大
        - ⚠️ 深度增加时，激活值累积
        
        **Transformer (ViT) 特点**：
        - ⚠️ Self-Attention内存消耗大（O(N²)）
        - ⚠️ 需要保存所有patch的特征
        - ✅ 参数量相对固定
        - ⚠️ Batch Size影响更明显
        
        **关键结论**：
        1. **参数量 ≠ 内存占用**
           - 100M参数的模型可能只占用200MB参数内存
           - 但激活值和梯度可能占用GB级内存
        
        2. **架构影响巨大**
           - CNN: 内存主要在激活值（与分辨率相关）
           - Transformer: 内存主要在attention矩阵（与序列长度²相关）
        
        3. **Batch Size的影响**
           - 所有内存组件都线性增长
           - Batch=32相比Batch=1，内存增加~32倍
        
        4. **训练 vs 推理**
           - 推理: 只需前向内存（约1/2-1/3）
           - 训练: 需要保存所有中间激活值和梯度
        """
        )

    # Batch Size影响实验
    st.markdown("---")
    st.subheader("📊 Batch Size对内存的影响")

    if st.button("🧪 运行Batch Size实验", key="batch_experiment"):
        with st.spinner("实验中..."):
            # 选择一个模型
            test_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 1000),
            )

            batch_sizes = [1, 2, 4, 8, 16, 32]
            memories = []

            for bs in batch_sizes:
                result = analyze_model_memory(
                    test_model, (bs, 3, 224, 224), detailed=False
                )
                memories.append(result["summary"]["peak_memory"])

        st.success("✅ 实验完成！")

        # 绘制曲线
        fig_batch = go.Figure()

        fig_batch.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=memories,
                mode="lines+markers",
                name="峰值内存",
                line=dict(color="red", width=3),
                marker=dict(size=10),
            )
        )

        # 添加线性参考线
        linear_ref = [memories[0] * bs for bs in batch_sizes]
        fig_batch.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=linear_ref,
                mode="lines",
                name="理论线性增长",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        fig_batch.update_layout(
            title="Batch Size vs 峰值内存",
            xaxis_title="Batch Size",
            yaxis_title="峰值内存 (MB)",
            height=400,
        )

        st.plotly_chart(fig_batch, use_container_width=True)

        st.info(
            f"""
        **实验结果**：
        - Batch=1: {memories[0]:.2f} MB
        - Batch=32: {memories[-1]:.2f} MB
        - 增长倍数: {memories[-1]/memories[0]:.1f}x
        
        **结论**: 内存与Batch Size **近似线性增长**
        
        **实用建议**:
        - 显存不够？先尝试减半Batch Size
        - 需要大Batch？使用梯度累积
        - 估算公式: 内存(BS=N) ≈ 内存(BS=1) × N
        """
        )

    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")

    st.markdown(
        """
    ### 内存分析的重要性
    
    1. **避免OOM错误**
       - Out Of Memory是训练中最常见的问题
       - 提前分析可以预测内存需求
       - 找到瓶颈层进行优化
    
    2. **优化训练策略**
       - 合理选择Batch Size
       - 决定是否使用梯度检查点
       - 选择混合精度训练
    
    3. **硬件选型**
       - 预测需要多大显存的GPU
       - 评估是否能在目标设备上运行
       - 云服务器成本估算
    
    ### 典型内存占用参考
    
    | 模型 | 输入尺寸 | Batch=1 | Batch=16 | Batch=32 |
    |------|---------|---------|----------|----------|
    | ResNet-50 | 224×224 | ~0.5GB | ~4GB | ~8GB |
    | ViT-Base | 224×224 | ~0.8GB | ~6GB | ~12GB |
    | BERT-Base | seq=512 | ~1GB | ~8GB | ~16GB |
    
    ### 快速优化建议
    
    - 内存不够？→ 降低Batch Size
    - 需要大Batch？→ 使用梯度累积
    - 仍然不够？→ 混合精度训练（FP16）
    - 还是不够？→ 梯度检查点（用时间换空间）
    """
    )


if __name__ == "__main__":
    # 测试运行
    memory_analysis_tab()
