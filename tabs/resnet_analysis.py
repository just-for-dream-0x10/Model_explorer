"""
ResNet残差连接分析
ResNet Residual Connection Analysis

验证残差连接如何解决梯度消失问题
核心理念：用数值证明"梯度高速公路"这个经典概念
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.resnet_models import (
    get_resnet_comparison, 
    TinyPlainNet, 
    TinyResNet
)


def analyze_gradient_flow(model, input_size, num_samples=10):
    """
    分析梯度流
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        num_samples: 采样次数
    
    Returns:
        gradient_stats: 梯度统计信息
    """
    model.train()
    gradient_norms = []
    layer_names = []
    
    # 收集所有可训练参数
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    
    for _ in range(num_samples):
        model.zero_grad()
        
        # 前向传播
        x = torch.randn(input_size)
        y = model(x)
        
        # 构造损失
        target = torch.randint(0, y.size(-1), (y.size(0),))
        loss = nn.CrossEntropyLoss()(y, target)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度范数
        if len(gradient_norms) == 0:
            for name, p in named_params:
                if p.grad is not None:
                    layer_names.append(name)
                    gradient_norms.append([])
        
        for i, (name, p) in enumerate(named_params):
            if p.grad is not None and i < len(gradient_norms):
                grad_norm = p.grad.norm().item()
                gradient_norms[i].append(grad_norm)
    
    # 计算统计量
    gradient_stats = []
    for i, norms in enumerate(gradient_norms):
        if norms and i < len(layer_names):
            gradient_stats.append({
                "layer": layer_names[i],
                "mean": np.mean(norms),
                "std": np.std(norms),
                "min": np.min(norms),
                "max": np.max(norms)
            })
    
    return gradient_stats


def plot_gradient_comparison(plain_stats, resnet_stats):
    """
    对比普通网络和ResNet的梯度流
    
    Args:
        plain_stats: 普通网络的梯度统计
        resnet_stats: ResNet的梯度统计
    
    Returns:
        fig: Plotly图表
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("普通网络（无残差）", "ResNet（有残差）"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 普通网络
    if plain_stats:
        plain_means = [stat["mean"] for stat in plain_stats]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(plain_means))),
                y=plain_means,
                mode='lines+markers',
                name='普通网络',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # ResNet
    if resnet_stats:
        resnet_means = [stat["mean"] for stat in resnet_stats]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(resnet_means))),
                y=resnet_means,
                mode='lines+markers',
                name='ResNet',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
    
    # 添加警戒线
    for col in [1, 2]:
        fig.add_hline(y=1e-5, line_dash="dash", line_color="orange", 
                      annotation_text="梯度消失警戒线", row=1, col=col)
    
    fig.update_xaxes(title_text="层索引", row=1, col=1)
    fig.update_xaxes(title_text="层索引", row=1, col=2)
    fig.update_yaxes(title_text="梯度范数（对数）", type="log", row=1, col=1)
    fig.update_yaxes(title_text="梯度范数（对数）", type="log", row=1, col=2)
    
    fig.update_layout(
        title="梯度流对比分析",
        height=500,
        showlegend=True
    )
    
    return fig


def plot_gradient_statistics(plain_stats, resnet_stats):
    """
    绘制梯度统计对比（箱线图风格）
    
    Args:
        plain_stats: 普通网络的梯度统计
        resnet_stats: ResNet的梯度统计
    
    Returns:
        fig: Plotly图表
    """
    # 计算平均梯度
    plain_avg = np.mean([stat["mean"] for stat in plain_stats]) if plain_stats else 0
    resnet_avg = np.mean([stat["mean"] for stat in resnet_stats]) if resnet_stats else 0
    
    # 计算梯度消失层数（梯度 < 1e-5）
    plain_vanished = sum(1 for stat in plain_stats if stat["mean"] < 1e-5)
    resnet_vanished = sum(1 for stat in resnet_stats if stat["mean"] < 1e-5)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['平均梯度范数', '梯度消失层数'],
        y=[plain_avg, plain_vanished],
        name='普通网络',
        marker_color='red',
        text=[f'{plain_avg:.2e}', f'{plain_vanished}'],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=['平均梯度范数', '梯度消失层数'],
        y=[resnet_avg, resnet_vanished],
        name='ResNet',
        marker_color='green',
        text=[f'{resnet_avg:.2e}', f'{resnet_vanished}'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="梯度统计对比",
        xaxis_title="指标",
        yaxis_title="数值",
        barmode='group',
        height=400
    )
    
    return fig


def explain_residual_math():
    """展示残差连接的数学原理"""
    st.markdown("""
    ### 📐 残差连接的数学原理
    
    #### 普通网络
    ```
    y = F(x)
    ```
    - 输出完全依赖于函数F的学习
    - 反向传播：∂L/∂x = ∂L/∂y · ∂F/∂x
    - 问题：∂F/∂x 可能很小，导致梯度消失
    
    #### ResNet（残差网络）
    ```
    y = F(x) + x
    ```
    - 输出 = 残差函数F(x) + 恒等映射x
    - 反向传播：∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
    - **关键**：即使∂F/∂x很小，由于"+1"项的存在，梯度仍能传播！
    
    #### 为什么"+1"很重要？
    
    假设有L层，梯度需要从第L层传到第1层：
    
    **普通网络**：
    ```
    ∂L/∂x₁ = ∂L/∂xₗ · ∂xₗ/∂xₗ₋₁ · ... · ∂x₂/∂x₁
    ```
    如果每层的梯度∂xᵢ/∂xᵢ₋₁ = 0.5，则：
    - 10层：0.5¹⁰ ≈ 0.001
    - 20层：0.5²⁰ ≈ 0.000001 ⚠️ **梯度消失！**
    
    **ResNet**：
    ```
    ∂L/∂x₁ = ∂L/∂xₗ · (∂F/∂xₗ₋₁ + 1) · ... · (∂F/∂x₁ + 1)
    ```
    即使∂F/∂xᵢ = 0，梯度仍能通过"+1"项传播：
    - 10层：至少保证梯度 = ∂L/∂xₗ · 1 · 1 · ... · 1 = ∂L/∂xₗ
    - ✅ **梯度高速公路**：绕过了梯度消失的障碍！
    """)


def resnet_analysis_tab(chinese_supported=True):
    """ResNet残差连接分析主函数"""
    
    st.header("🏗️ ResNet残差连接分析")
    st.markdown("""
    > **核心问题**：残差连接（Residual Connection）如何解决深度网络的梯度消失问题？
    
    **验证方法**：对比相同深度的普通网络和ResNet，观察梯度流的真实差异
    """)
    
    st.markdown("---")
    
    # 数学原理
    with st.expander("📐 数学原理（点击展开）", expanded=False):
        explain_residual_math()
    
    st.markdown("---")
    
    # 网络选择
    st.subheader("🔧 实验配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        network_type = st.selectbox(
            "选择网络类型",
            ["简化版（全连接，快速）", "完整版（卷积，真实）"],
            help="简化版用于快速演示，完整版更接近实际ResNet"
        )
    
    with col2:
        if network_type == "简化版（全连接，快速）":
            num_layers = st.slider("网络深度（层数）", 10, 50, 20, 5)
        else:
            num_blocks = st.slider("残差块数量", 5, 20, 10, 5)
    
    # 构建网络
    st.markdown("---")
    st.subheader("🏗️ 网络结构")
    
    if network_type == "简化版（全连接，快速）":
        plain_net = TinyPlainNet(num_layers=num_layers, hidden_dim=128)
        resnet = TinyResNet(num_layers=num_layers, hidden_dim=128)
        input_size = (8, 10)  # Batch size = 8
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**普通网络结构**")
            st.code(f"""
输入层: Linear(10 -> 128) + ReLU
隐藏层: {num_layers - 2}层 Linear(128 -> 128) + ReLU
输出层: Linear(128 -> 2)

特点: 无残差连接
            """)
        
        with col2:
            st.markdown("**ResNet结构**")
            st.code(f"""
输入层: Linear(10 -> 128) + ReLU
残差块: {(num_layers - 2) // 2}个
  每个块: Linear -> ReLU -> Linear
          + 残差连接 (y = F(x) + x)
输出层: Linear(128 -> 2)

特点: 有残差连接
            """)
    
    else:
        plain_net, resnet, info = get_resnet_comparison(num_blocks=num_blocks)
        input_size = (8, 3, 224, 224)  # Batch size = 8
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**普通网络结构**")
            st.code(f"""
初始层: Conv2d(3->64, k=7, s=2) + BN + MaxPool
主体: {num_blocks}个普通块
  每个块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
分类头: AdaptiveAvgPool + Linear

参数量: {info['plain_params']:,}
            """)
        
        with col2:
            st.markdown("**ResNet结构**")
            st.code(f"""
初始层: Conv2d(3->64, k=7, s=2) + BN + MaxPool
主体: {num_blocks}个残差块
  每个块: Conv -> BN -> ReLU -> Conv -> BN
          + 残差连接 (y = F(x) + x) + ReLU
分类头: AdaptiveAvgPool + Linear

参数量: {info['resnet_params']:,}
            """)
    
    # 梯度流分析
    st.markdown("---")
    st.subheader("🔬 梯度流分析")
    
    st.info("💡 点击下方按钮，模拟训练过程并分析梯度流")
    
    if st.button("🚀 开始梯度分析", type="primary"):
        
        with st.spinner("分析中...这可能需要几秒钟"):
            # 分析普通网络
            st.write("分析普通网络...")
            plain_stats = analyze_gradient_flow(plain_net, input_size, num_samples=5)
            
            # 分析ResNet
            st.write("分析ResNet...")
            resnet_stats = analyze_gradient_flow(resnet, input_size, num_samples=5)
        
        st.success("✅ 分析完成！")
        
        # 显示关键指标
        st.markdown("#### 📊 关键指标对比")
        
        col1, col2, col3, col4 = st.columns(4)
        
        plain_avg = np.mean([stat["mean"] for stat in plain_stats]) if plain_stats else 0
        resnet_avg = np.mean([stat["mean"] for stat in resnet_stats]) if resnet_stats else 0
        plain_vanished = sum(1 for stat in plain_stats if stat["mean"] < 1e-5)
        resnet_vanished = sum(1 for stat in resnet_stats if stat["mean"] < 1e-5)
        
        with col1:
            st.metric("普通网络平均梯度", f"{plain_avg:.2e}")
        with col2:
            st.metric("ResNet平均梯度", f"{resnet_avg:.2e}", 
                     delta=f"{((resnet_avg/plain_avg - 1) * 100):.1f}%" if plain_avg > 0 else None)
        with col3:
            st.metric("普通网络梯度消失层数", plain_vanished)
        with col4:
            st.metric("ResNet梯度消失层数", resnet_vanished,
                     delta=f"{resnet_vanished - plain_vanished}" if plain_vanished > 0 else None,
                     delta_color="inverse")
        
        # 梯度流对比图
        st.markdown("#### 📈 梯度流可视化")
        fig1 = plot_gradient_comparison(plain_stats, resnet_stats)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 统计对比图
        st.markdown("#### 📊 统计对比")
        fig2 = plot_gradient_statistics(plain_stats, resnet_stats)
        st.plotly_chart(fig2, use_container_width=True)
        
        # 详细分析
        st.markdown("#### 🔍 详细分析")
        
        # 只显示前10层和后10层
        st.markdown("**普通网络梯度详情（前10层 + 后10层）**")
        display_plain = plain_stats[:10] + plain_stats[-10:] if len(plain_stats) > 20 else plain_stats
        
        for i, stat in enumerate(display_plain):
            mean_grad = stat['mean']
            if mean_grad < 1e-5:
                st.error(f"❌ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (严重消失！)")
            elif mean_grad < 1e-3:
                st.warning(f"⚠️ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (轻微消失)")
            else:
                st.success(f"✅ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (正常)")
        
        st.markdown("**ResNet梯度详情（前10层 + 后10层）**")
        display_resnet = resnet_stats[:10] + resnet_stats[-10:] if len(resnet_stats) > 20 else resnet_stats
        
        for i, stat in enumerate(display_resnet):
            mean_grad = stat['mean']
            if mean_grad < 1e-5:
                st.error(f"❌ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (严重消失！)")
            elif mean_grad < 1e-3:
                st.warning(f"⚠️ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (轻微消失)")
            else:
                st.success(f"✅ 层 {i+1}: {stat['layer'][:50]} | 梯度={mean_grad:.2e} (正常)")
        
        # 结论
        st.markdown("---")
        st.subheader("📚 实验结论")
        
        if resnet_avg > plain_avg * 1.5:
            st.success(f"""
            ✅ **残差连接显著改善了梯度流！**
            
            - ResNet的平均梯度是普通网络的 **{resnet_avg/plain_avg:.1f}倍**
            - 梯度消失层数从 {plain_vanished} 层减少到 {resnet_vanished} 层
            - 证明了"梯度高速公路"机制的有效性
            
            **关键原因**：y = F(x) + x 中的"+x"项确保了梯度至少能以恒等映射的方式传播
            """)
        else:
            st.info(f"""
            ℹ️ **当前配置下差异不明显**
            
            可能原因：
            1. 网络深度不够（建议增加到30层以上）
            2. 使用了BatchNorm（已经缓解了梯度消失）
            3. 采样次数较少（统计噪声）
            
            建议：增加网络深度或去除BatchNorm后重新测试
            """)
    
    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")
    
    st.markdown("""
    1. **残差连接不是魔法，而是数学**
       - 公式：y = F(x) + x
       - 梯度：∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
       - 关键："+1"项确保梯度能传播
    
    2. **为什么叫"高速公路"？**
       - 普通网络：梯度必须经过每一层的复杂变换
       - ResNet：梯度可以通过"+x"直接传播（跳过变换）
       - 就像高速公路可以绕过城市拥堵
    
    3. **实际工程意义**
       - 可以训练非常深的网络（100+层）
       - 梯度更稳定，训练更容易
       - 现代架构的标配（Transformer也用残差连接）
    
    4. **何时需要残差连接？**
       - 网络深度 > 20层
       - 出现梯度消失/梯度爆炸
       - 需要训练极深的模型
    """)


if __name__ == "__main__":
    # 测试运行
    resnet_analysis_tab()
