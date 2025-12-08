"""
归一化层对比工具
Normalization Layer Comparison Tool

对比不同归一化方法的效果
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from utils.visualization.chart_utils import format_number


def normalization_comparison_tab(CHINESE_SUPPORTED):
    """归一化层对比标签页内容"""

    st.header(
        "📏 归一化层对比" if CHINESE_SUPPORTED else "📏 Normalization Layer Comparison"
    )

    st.markdown(
        """
    对比三种主要归一化方法的效果：
    - **BatchNorm**: 在批次维度归一化
    - **LayerNorm**: 在特征维度归一化  
    - **GroupNorm**: 在组内归一化
    """
    )

    # 参数控制
    col1, col2 = st.columns(2)

    with col1:
        num_channels = st.slider(
            "通道数" if CHINESE_SUPPORTED else "Number of Channels",
            min_value=4,
            max_value=128,
            value=16,
            step=4,
            key="norm_channels",
        )

    with col2:
        batch_size = st.slider(
            "批次大小" if CHINESE_SUPPORTED else "Batch Size",
            min_value=4,
            max_value=64,
            value=32,
            step=4,
            key="norm_batch_size",
        )

    # 简化的归一化对比
    if st.button("🚀 生成数据并对比", type="primary"):
        with st.spinner("生成数据并计算..."):
            # 创建测试数据
            torch.manual_seed(42)
            spatial_size = 16  # 保持较小值以确保性能

            x = (
                torch.randn(batch_size, num_channels, spatial_size, spatial_size) * 10
                + 5
            )

            # 显示原始数据统计
            st.info(
                f"""
            **原始数据统计**：
            - 形状: [{batch_size}, {num_channels}, {spatial_size}, {spatial_size}]
            - 均值: {x.mean():.4f}
            - 标准差: {x.std():.4f}
            - 范围: [{x.min():.4f}, {x.max():.4f}]
            """
            )

            # 简化的归一化计算
            # BatchNorm
            batch_norm_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_norm_std = x.std(dim=[0, 2, 3], keepdim=True)
            batch_norm = (x - batch_norm_mean) / (batch_norm_std + 1e-5)

            # LayerNorm
            layer_norm_mean = x.mean(dim=-1, keepdim=True)
            layer_norm_std = x.std(dim=-1, keepdim=True)
            layer_norm = (x - layer_norm_mean) / (layer_norm_std + 1e-5)

            # 显示关键指标
            st.markdown("#### 📈 归一化后的统计量")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("原始均值", f"{x.mean():.4f}")
                st.metric("原始标准差", f"{x.std():.4f}")

            with col2:
                st.metric("BatchNorm均值", f"{batch_norm.mean():.6f}")
                st.metric("BatchNorm标准差", f"{batch_norm.std():.6f}")

            with col3:
                st.metric("LayerNorm均值", f"{layer_norm.mean():.6f}")
                st.metric("LayerNorm标准差", f"{layer_norm.std():.6f}")

            with col4:
                st.metric("数据形状", f"{x.shape}")

            st.success("✅ 观察：归一化后，均值≈0、标准差≈1")

            # 简化的可视化
            st.markdown("#### 📊 激活值分布")

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=x.flatten().numpy(), name="原始数据", opacity=0.7, nbinsx=50
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=batch_norm.flatten().numpy(),
                    name="BatchNorm",
                    opacity=0.7,
                    nbinsx=50,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=layer_norm.flatten().numpy(),
                    name="LayerNorm",
                    opacity=0.7,
                    nbinsx=50,
                )
            )

            fig.update_layout(
                title="激活值分布对比",
                xaxis_title="激活值",
                yaxis_title="频次",
                barmode="overlay",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        **关键观察**：
        - **BatchNorm**: 在batch维度归一化，适合CNN
        - **LayerNorm**: 在特征维度归一化，适合Transformer
        - 所有方法都将数据调整到均值≈0、标准差≈1
        """
    )
