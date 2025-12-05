"""
归一化层对比分析
Normalization Layers Comparison

对比BatchNorm、LayerNorm、GroupNorm的工作机制和适用场景
核心理念：用可视化展示"在哪个维度归一化"的差异
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.normalization_layers import (
    apply_batch_norm,
    apply_layer_norm,
    apply_group_norm,
    compare_normalization_methods,
    SimpleCNNWithNorm,
    get_normalization_comparison_info
)


def plot_activation_distribution(data, title="激活值分布"):
    """
    绘制激活值分布直方图
    
    Args:
        data: 激活值数据
        title: 图表标题
    
    Returns:
        fig: Plotly图表
    """
    fig = go.Figure()
    
    # 采样以加快绘制速度
    sample_size = min(10000, data.size)
    sampled_data = np.random.choice(data.flatten(), size=sample_size, replace=False)
    
    fig.add_trace(go.Histogram(
        x=sampled_data,
        nbinsx=30,  # 从50减少到30
        name='激活值',
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="激活值",
        yaxis_title="频数",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_normalization_comparison(original, batch_norm, layer_norm, group_norm):
    """
    对比四种情况的激活值分布
    
    Args:
        original: 原始激活值
        batch_norm: BatchNorm后的激活值
        layer_norm: LayerNorm后的激活值
        group_norm: GroupNorm后的激活值
    
    Returns:
        fig: Plotly图表
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("原始激活值", "BatchNorm", "LayerNorm", "GroupNorm")
    )
    
    # 采样数据以加快渲染
    sample_size = 5000
    
    # 原始
    orig_sample = np.random.choice(original.flatten(), size=min(sample_size, original.size), replace=False)
    fig.add_trace(
        go.Histogram(x=orig_sample, nbinsx=30, name='原始', 
                    marker_color='gray', opacity=0.7),
        row=1, col=1
    )
    
    # BatchNorm
    bn_sample = np.random.choice(batch_norm.flatten(), size=min(sample_size, batch_norm.size), replace=False)
    fig.add_trace(
        go.Histogram(x=bn_sample, nbinsx=30, name='BatchNorm',
                    marker_color='red', opacity=0.7),
        row=1, col=2
    )
    
    # LayerNorm
    ln_sample = np.random.choice(layer_norm.flatten(), size=min(sample_size, layer_norm.size), replace=False)
    fig.add_trace(
        go.Histogram(x=ln_sample, nbinsx=30, name='LayerNorm',
                    marker_color='green', opacity=0.7),
        row=2, col=1
    )
    
    # GroupNorm
    gn_sample = np.random.choice(group_norm.flatten(), size=min(sample_size, group_norm.size), replace=False)
    fig.add_trace(
        go.Histogram(x=gn_sample, nbinsx=30, name='GroupNorm',
                    marker_color='blue', opacity=0.7),
        row=2, col=2
    )
    
    fig.update_layout(
        title="归一化方法对比",
        height=600,
        showlegend=False
    )
    
    return fig


def plot_channel_statistics(original, batch_norm, layer_norm, group_norm, num_channels=16):
    """
    绘制每个通道的均值和标准差
    
    Args:
        original: 原始激活值 [B, C, H, W]
        batch_norm: BatchNorm后 [B, C, H, W]
        layer_norm: LayerNorm后 [B, C, H, W]
        group_norm: GroupNorm后 [B, C, H, W]
        num_channels: 显示前N个通道
    
    Returns:
        fig: Plotly图表
    """
    # 计算每个通道的统计量
    def compute_channel_stats(x):
        # x: [B, C, H, W]
        means = x.mean(dim=[0, 2, 3]).detach().cpu().numpy()[:num_channels]
        stds = x.std(dim=[0, 2, 3]).detach().cpu().numpy()[:num_channels]
        return means, stds
    
    orig_means, orig_stds = compute_channel_stats(original)
    bn_means, bn_stds = compute_channel_stats(batch_norm)
    ln_means, ln_stds = compute_channel_stats(layer_norm)
    gn_means, gn_stds = compute_channel_stats(group_norm)
    
    channels = list(range(num_channels))
    
    # 均值对比
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("各通道均值对比", "各通道标准差对比")
    )
    
    # 均值
    fig.add_trace(
        go.Bar(x=channels, y=orig_means, name='原始', marker_color='gray'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=channels, y=bn_means, name='BatchNorm', marker_color='red'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=channels, y=ln_means, name='LayerNorm', marker_color='green'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=channels, y=gn_means, name='GroupNorm', marker_color='blue'),
        row=1, col=1
    )
    
    # 标准差
    fig.add_trace(
        go.Bar(x=channels, y=orig_stds, name='原始', marker_color='gray', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=channels, y=bn_stds, name='BatchNorm', marker_color='red', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=channels, y=ln_stds, name='LayerNorm', marker_color='green', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=channels, y=gn_stds, name='GroupNorm', marker_color='blue', showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="通道索引", row=1, col=1)
    fig.update_xaxes(title_text="通道索引", row=1, col=2)
    fig.update_yaxes(title_text="均值", row=1, col=1)
    fig.update_yaxes(title_text="标准差", row=1, col=2)
    
    fig.update_layout(
        title="通道统计量对比",
        height=400,
        barmode='group'
    )
    
    return fig


def explain_normalization_math():
    """展示归一化的数学原理"""
    st.markdown("""
    ### 📐 归一化层的数学原理
    
    归一化的核心思想：**将激活值调整到均值为0、方差为1的分布**
    
    #### 通用公式
    ```
    y = γ · (x - μ) / √(σ² + ε) + β
    ```
    - μ: 均值
    - σ²: 方差
    - ε: 数值稳定项（防止除以0）
    - γ, β: 可学习的缩放和平移参数
    
    #### 关键差异：在哪个维度计算μ和σ²？
    
    假设输入形状为 `[Batch, Channel, Height, Width]` = `[B, C, H, W]`
    
    **1. BatchNorm**
    ```
    μ = mean(x, dim=[B, H, W])  # 对每个通道，在batch和空间维度求均值
    σ² = var(x, dim=[B, H, W])
    
    结果形状: μ和σ²都是 [C] 维
    ```
    
    **2. LayerNorm**
    ```
    μ = mean(x, dim=[C, H, W])  # 对每个样本，在通道和空间维度求均值
    σ² = var(x, dim=[C, H, W])
    
    结果形状: μ和σ²都是 [B] 维
    ```
    
    **3. GroupNorm**
    ```
    先将C分成G组，每组有C/G个通道
    μ = mean(x, dim=[C/G, H, W])  # 对每个样本的每组，在组内求均值
    σ² = var(x, dim=[C/G, H, W])
    
    结果形状: μ和σ²都是 [B, G] 维
    ```
    
    #### 直观理解
    
    | 方法 | 归一化维度 | 通俗解释 |
    |------|-----------|---------|
    | **BatchNorm** | 跨batch归一化 | 同一个通道，看所有样本的统计量 |
    | **LayerNorm** | 跨通道归一化 | 同一个样本，看所有通道的统计量 |
    | **GroupNorm** | 分组归一化 | 同一个样本，将通道分组后归一化 |
    """)


def normalization_comparison_tab(chinese_supported=True):
    """归一化层对比分析主函数"""
    
    st.header("🔧 归一化层对比分析")
    st.markdown("""
    > **核心问题**：BatchNorm、LayerNorm、GroupNorm有什么区别？分别适用于什么场景？
    
    **验证方法**：用可视化展示"在哪个维度归一化"的实际效果
    """)
    
    st.markdown("---")
    
    # 数学原理
    with st.expander("📐 数学原理（点击展开）", expanded=False):
        explain_normalization_math()
    
    st.markdown("---")
    
    # 交互式演示
    st.subheader("🎨 交互式可视化")
    
    st.info("💡 提示：为了加快加载速度，使用较小的数据规模进行演示")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batch_size = st.slider("Batch Size", 1, 8, 4, 1,
                              help="批次大小，BatchNorm依赖此参数")
    
    with col2:
        num_channels = st.slider("通道数", 16, 64, 32, 16,
                                help="特征通道数")
    
    with col3:
        num_groups = st.slider("GroupNorm分组数", 4, 16, 8, 4,
                              help="GroupNorm的分组数量")
    
    # 生成测试数据
    st.markdown("---")
    st.subheader("📊 激活值分布对比")
    
    if st.button("🚀 生成数据并对比", type="primary"):
        with st.spinner("生成数据并计算..."):
            # 创建测试数据（使用更小的空间尺寸以加快计算）
            torch.manual_seed(42)
            spatial_size = 16  # 从32改为16，减少75%的计算量
            x = torch.randn(batch_size, num_channels, spatial_size, spatial_size) * 10 + 5
            
            # 显示原始数据统计
            st.info(f"""
            **原始数据统计**：
            - 形状: [{batch_size}, {num_channels}, {spatial_size}, {spatial_size}]
            - 均值: {x.mean():.4f}
            - 标准差: {x.std():.4f}
            - 范围: [{x.min():.4f}, {x.max():.4f}]
            """)
            
            # 对比三种归一化方法
            results = compare_normalization_methods(x, num_groups=num_groups)
            
            # 提取归一化后的数据
            original = x
            batch_norm = results["batch_norm"]["normalized"]
            layer_norm = results["layer_norm"]["normalized"]
            group_norm = results["group_norm"]["normalized"]
            
            # 显示关键指标
            st.markdown("#### 📈 归一化后的统计量")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("原始均值", f"{original.mean():.4f}")
                st.metric("原始标准差", f"{original.std():.4f}")
            
            with col2:
                bn_stats = results["batch_norm"]["stats"]
                st.metric("BatchNorm均值", f"{bn_stats['normalized_mean']:.6f}")
                st.metric("BatchNorm标准差", f"{bn_stats['normalized_std']:.6f}")
            
            with col3:
                ln_stats = results["layer_norm"]["stats"]
                st.metric("LayerNorm均值", f"{ln_stats['normalized_mean']:.6f}")
                st.metric("LayerNorm标准差", f"{ln_stats['normalized_std']:.6f}")
            
            with col4:
                gn_stats = results["group_norm"]["stats"]
                st.metric("GroupNorm均值", f"{gn_stats['normalized_mean']:.6f}")
                st.metric("GroupNorm标准差", f"{gn_stats['normalized_std']:.6f}")
            
            # 可视化：分布对比
            st.markdown("#### 📊 激活值分布直方图")
            fig1 = plot_normalization_comparison(
                original.detach().cpu().numpy(),
                batch_norm.detach().cpu().numpy(),
                layer_norm.detach().cpu().numpy(),
                group_norm.detach().cpu().numpy()
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.success("✅ 观察：归一化后，所有方法都将激活值分布调整到均值≈0、标准差≈1")
            
            # 可视化：通道统计
            if num_channels >= 16:
                st.markdown("#### 📊 各通道统计量对比")
                fig2 = plot_channel_statistics(
                    original, batch_norm, layer_norm, group_norm,
                    num_channels=min(16, num_channels)
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                st.info("""
                **关键观察**：
                - **BatchNorm**: 每个通道的均值≈0、标准差≈1（在batch维度归一化）
                - **LayerNorm**: 不同通道的均值和标准差不同（在通道维度归一化）
                - **GroupNorm**: 介于两者之间
                """)
    
    # 适用场景对比
    st.markdown("---")
    st.subheader("🎯 适用场景对比")
    
    info = get_normalization_comparison_info()
    
    tab1, tab2, tab3 = st.tabs(["BatchNorm", "LayerNorm", "GroupNorm"])
    
    with tab1:
        bn_info = info["batch_norm"]
        st.markdown(f"### {bn_info['name']}")
        st.code(bn_info['formula'])
        st.markdown(f"**归一化维度**: {bn_info['normalization_dim']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ 优势**")
            for adv in bn_info['advantages']:
                st.markdown(f"- {adv}")
        
        with col2:
            st.markdown("**❌ 劣势**")
            for dis in bn_info['disadvantages']:
                st.markdown(f"- {dis}")
        
        st.markdown("**🎯 何时使用**")
        for use in bn_info['when_to_use']:
            st.markdown(f"- {use}")
    
    with tab2:
        ln_info = info["layer_norm"]
        st.markdown(f"### {ln_info['name']}")
        st.code(ln_info['formula'])
        st.markdown(f"**归一化维度**: {ln_info['normalization_dim']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ 优势**")
            for adv in ln_info['advantages']:
                st.markdown(f"- {adv}")
        
        with col2:
            st.markdown("**❌ 劣势**")
            for dis in ln_info['disadvantages']:
                st.markdown(f"- {dis}")
        
        st.markdown("**🎯 何时使用**")
        for use in ln_info['when_to_use']:
            st.markdown(f"- {use}")
    
    with tab3:
        gn_info = info["group_norm"]
        st.markdown(f"### {gn_info['name']}")
        st.code(gn_info['formula'])
        st.markdown(f"**归一化维度**: {gn_info['normalization_dim']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ 优势**")
            for adv in gn_info['advantages']:
                st.markdown(f"- {adv}")
        
        with col2:
            st.markdown("**❌ 劣势**")
            for dis in gn_info['disadvantages']:
                st.markdown(f"- {dis}")
        
        st.markdown("**🎯 何时使用**")
        for use in gn_info['when_to_use']:
            st.markdown(f"- {use}")
    
    # Batch Size敏感性分析
    st.markdown("---")
    st.subheader("🔬 Batch Size敏感性分析")
    
    st.markdown("""
    **关键问题**：为什么BatchNorm在小batch场景下效果差？
    
    让我们用实验证明：
    """)
    
    if st.button("🧪 运行Batch Size敏感性测试"):
        with st.spinner("测试中..."):
            batch_sizes = [1, 2, 4, 8, 16, 32]
            bn_stds = []
            ln_stds = []
            gn_stds = []
            
            for bs in batch_sizes:
                x = torch.randn(bs, 64, 32, 32) * 10 + 5
                
                bn_normalized, bn_stats = apply_batch_norm(x)
                ln_normalized, ln_stats = apply_layer_norm(x)
                gn_normalized, gn_stats = apply_group_norm(x, num_groups=32)
                
                bn_stds.append(bn_stats['normalized_std'])
                ln_stds.append(ln_stats['normalized_std'])
                gn_stds.append(gn_stats['normalized_std'])
            
            # 绘制图表
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=batch_sizes, y=bn_stds,
                mode='lines+markers',
                name='BatchNorm',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=batch_sizes, y=ln_stds,
                mode='lines+markers',
                name='LayerNorm',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=batch_sizes, y=gn_stds,
                mode='lines+markers',
                name='GroupNorm',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="理想值=1.0")
            
            fig.update_layout(
                title="不同Batch Size下的归一化效果",
                xaxis_title="Batch Size",
                yaxis_title="归一化后标准差",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("""
            ✅ **实验结论**：
            - **BatchNorm**: 在小batch时标准差偏离1.0较大（统计不准确）
            - **LayerNorm/GroupNorm**: 与batch size无关，始终稳定
            - 这就是为什么Transformer使用LayerNorm而不是BatchNorm！
            """)
    
    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")
    
    st.markdown("""
    ### 快速选择指南
    
    | 场景 | 推荐方法 | 原因 |
    |------|---------|------|
    | **CNN + 大batch** | BatchNorm | 加速收敛，效果最好 |
    | **Transformer** | LayerNorm | 与batch无关，适合序列 |
    | **目标检测（小batch）** | GroupNorm | BatchNorm效果差时的替代 |
    | **RNN/LSTM** | LayerNorm | 序列长度可变 |
    | **在线学习** | LayerNorm/GroupNorm | 不依赖batch统计 |
    
    ### 记住三个关键差异
    
    1. **归一化维度不同**
       - BatchNorm: 跨batch归一化（依赖batch统计）
       - LayerNorm: 跨通道归一化（独立于batch）
       - GroupNorm: 分组归一化（折中方案）
    
    2. **Batch Size敏感性**
       - BatchNorm: 高度敏感（小batch效果差）
       - LayerNorm/GroupNorm: 不敏感
    
    3. **历史地位**
       - BatchNorm: CNN时代的标配（2015）
       - LayerNorm: Transformer时代的标配（2016）
       - GroupNorm: 小batch场景的救星（2018）
    """)


if __name__ == "__main__":
    # 测试运行
    normalization_comparison_tab()
