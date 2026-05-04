"""
失败案例博物馆 - Failure Cases Museum
展示常见的神经网络设计错误，帮助理解为什么某些设计会失败

核心理念：不是告诉你"这样不好"，而是让你看到"到底哪里出问题了"
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.failure_cases import get_failure_case


def calculate_params_and_memory(model, input_size, skip_forward=False):
    """
    计算模型的参数量和内存占用

    Args:
        model: PyTorch模型
        input_size: 输入尺寸 (tuple)
        skip_forward: 是否跳过前向传播（用于超大模型）

    Returns:
        dict: 包含参数量、内存等信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 估算前向传播内存（MB）
    # 假设float32，每个参数4字节
    param_memory = total_params * 4 / (1024**2)

    # 估算激活值内存（简化计算）
    activation_memory = 0.0
    if not skip_forward:
        try:
            x = torch.randn(input_size)
            with torch.no_grad():
                y = model(x)
            activation_memory = np.prod(y.shape) * 4 / (1024**2)
        except Exception:
            activation_memory = 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_memory,
        "activation_memory_mb": activation_memory,
        "total_memory_mb": param_memory + activation_memory,
    }


def simulate_gradient_flow(model, input_size, num_samples=10):
    """
    模拟梯度流，检测梯度消失/爆炸

    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        num_samples: 采样次数

    Returns:
        dict: 梯度统计信息
    """
    model.train()
    gradient_norms = []
    layer_names = []

    # 收集可训练参数
    named_params = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]

    for _ in range(num_samples):
        model.zero_grad()

        # 前向传播
        x = torch.randn(input_size)
        try:
            y = model(x)

            # 构造损失（简单的L2损失）
            target = torch.randn_like(y)
            loss = ((y - target) ** 2).mean()

            # 反向传播
            loss.backward()

            # 收集梯度范数
            if len(gradient_norms) == 0:
                for name, p in named_params:
                    if p.grad is not None:
                        layer_names.append(name)
                        gradient_norms.append([])

            for i, (name, p) in enumerate(named_params):
                if p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    if i < len(gradient_norms):
                        gradient_norms[i].append(grad_norm)
        except Exception as e:
            st.warning(f"梯度模拟失败: {e}")
            break

    # 计算统计量
    gradient_stats = []
    for i, norms in enumerate(gradient_norms):
        if norms:
            gradient_stats.append(
                {
                    "layer": layer_names[i] if i < len(layer_names) else f"layer_{i}",
                    "mean": np.mean(norms),
                    "std": np.std(norms),
                    "min": np.min(norms),
                    "max": np.max(norms),
                }
            )

    return gradient_stats


def simulate_training_with_lr(model, input_size, lr, num_steps=50):
    """
    模拟训练过程，观察不同学习率的影响

    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        lr: 学习率
        num_steps: 训练步数

    Returns:
        list: 每步的loss值
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # 生成随机数据
        x = torch.randn(input_size)
        target = torch.randn(model(x).shape)

        # 前向传播
        try:
            y = model(x)
            loss = criterion(y, target)

            # 检查是否为NaN
            if torch.isnan(loss):
                losses.append(float("nan"))
                st.warning(f"⚠️ Loss在第{step+1}步变成NaN！")
                break

            # 反向传播
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        except Exception as e:
            st.error(f"训练失败: {e}")
            break

    return losses


def plot_gradient_flow(gradient_stats):
    """绘制梯度流图"""
    if not gradient_stats:
        return None

    fig = go.Figure()

    layers = [stat["layer"] for stat in gradient_stats]
    means = [stat["mean"] for stat in gradient_stats]

    # 使用对数坐标
    fig.add_trace(
        go.Scatter(
            x=list(range(len(layers))),
            y=means,
            mode="lines+markers",
            name="梯度范数",
            line=dict(color="red", width=2),
            marker=dict(size=8),
        )
    )

    # 添加警戒线
    fig.add_hline(
        y=1e-5, line_dash="dash", line_color="orange", annotation_text="梯度消失警戒线"
    )
    fig.add_hline(
        y=1e2, line_dash="dash", line_color="purple", annotation_text="梯度爆炸警戒线"
    )

    fig.update_layout(
        title="梯度流分析（对数坐标）",
        xaxis_title="层索引",
        yaxis_title="梯度范数（对数）",
        yaxis_type="log",
        height=400,
        showlegend=True,
    )

    return fig


def plot_loss_curve(losses, title="Loss曲线"):
    """绘制Loss曲线"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines+markers",
            name="Loss",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title=title, xaxis_title="训练步数", yaxis_title="Loss", height=400
    )

    return fig


def failure_museum_tab(chinese_supported=True):
    """失败案例博物馆主函数"""

    st.header("🏛️ 失败案例博物馆")
    st.markdown(
        """
    > **教学目标**：通过实际案例，让你看到网络设计错误会导致什么具体的数值问题
    
    **核心理念**：不是告诉你"这样不好"，而是让你看到"梯度真的变成0了"、"参数真的有32亿个"
    """
    )

    st.markdown("---")

    # 案例选择
    st.subheader("📋 选择失败案例")

    case_options = {
        "100层普通MLP（梯度消失）": "deep_mlp",
        "卷积层直接接超大全连接（参数爆炸）": "conv_fc",
        "20层卷积网络无归一化（训练不稳定）": "no_norm",
        "简单MLP + 超大学习率（梯度爆炸）": "huge_lr",
    }

    selected_case_name = st.selectbox(
        "选择案例", list(case_options.keys()), help="选择一个经典的设计错误案例"
    )

    case_id = case_options[selected_case_name]

    # 加载案例
    try:
        model, case_info = get_failure_case(case_id)
    except Exception as e:
        st.error(f"加载案例失败: {e}")
        return

    # 显示案例信息
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📌 {case_info['name']}")
        st.markdown(f"**问题类型**: {case_info['problem']}")
        st.markdown(f"**症状**: {case_info['symptom']}")

    with col2:
        st.markdown("### 🔍 原因分析")
        st.markdown(case_info["reason"])
        st.markdown("### ✅ 解决方案")
        st.success(case_info["solution"])

    # 计算参数量和内存
    st.markdown("---")
    st.subheader("📊 参数量与内存分析")

    with st.spinner("计算中..."):
        skip_fwd = case_info.get("skip_forward", False)
        stats = calculate_params_and_memory(model, case_info["input_size"], skip_forward=skip_fwd)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总参数量", f"{stats['total_params']:,}")
    with col2:
        st.metric("参数内存", f"{stats['param_memory_mb']:.2f} MB")
    with col3:
        st.metric("激活值内存", f"{stats['activation_memory_mb']:.2f} MB")
    with col4:
        total_mem = stats["total_memory_mb"]
        mem_color = "normal" if total_mem < 1000 else "inverse"
        st.metric("总内存", f"{total_mem:.2f} MB", delta_color=mem_color)

    # 根据案例类型显示不同的诊断
    st.markdown("---")

    if case_id == "deep_mlp":
        st.subheader("🔬 梯度消失诊断")
        st.markdown("模拟10次前向+反向传播，观察各层的梯度范数")

        if st.button("🚀 开始梯度分析", key="grad_analysis"):
            with st.spinner("分析中..."):
                gradient_stats = simulate_gradient_flow(model, case_info["input_size"])

            if gradient_stats:
                # 显示梯度表格
                st.markdown("#### 各层梯度统计")

                # 只显示前10层和后10层
                if len(gradient_stats) > 20:
                    display_stats = gradient_stats[:10] + gradient_stats[-10:]
                    st.info("显示前10层和后10层的梯度统计")
                else:
                    display_stats = gradient_stats

                for i, stat in enumerate(display_stats):
                    mean_grad = stat["mean"]
                    if mean_grad < 1e-5:
                        st.error(
                            f"❌ {stat['layer']}: 梯度={mean_grad:.2e} (严重消失！)"
                        )
                    elif mean_grad < 1e-3:
                        st.warning(
                            f"⚠️ {stat['layer']}: 梯度={mean_grad:.2e} (轻微消失)"
                        )
                    else:
                        st.success(f"✅ {stat['layer']}: 梯度={mean_grad:.2e} (正常)")

                # 绘制梯度流图
                fig = plot_gradient_flow(gradient_stats)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    elif case_id == "conv_fc":
        st.subheader("💥 参数爆炸警告")

        # 计算全连接层的参数量
        fc_params = 64 * 224 * 224 * 1000
        st.error(f"⚠️ 全连接层参数量: **{fc_params:,}** (32亿参数！)")

        st.markdown("#### 为什么这么多？")
        st.code(
            """
输入特征图: [Batch, 64, 224, 224]
Flatten后:  [Batch, 64×224×224] = [Batch, 3,211,264]
全连接层:   Linear(3,211,264 -> 1000)
参数量 = 3,211,264 × 1000 + 1000 = 3,211,265,000
        """
        )

        st.markdown("#### ✅ 正确做法：全局平均池化")
        st.code(
            """
输入特征图: [Batch, 64, 224, 224]
全局平均池化: [Batch, 64, 1, 1] -> [Batch, 64]
全连接层:   Linear(64 -> 1000)
参数量 = 64 × 1000 + 1000 = 65,000 (减少了5万倍！)
        """
        )

    elif case_id == "no_norm":
        st.subheader("📉 训练不稳定模拟")
        st.markdown("比较有无BatchNorm的训练曲线差异")

        st.info(
            "💡 提示：由于时间限制，这里展示理论分析。实际训练可以看到Loss剧烈震荡。"
        )

        # 显示激活值分布分析
        st.markdown("#### 激活值分布问题")
        st.markdown(
            """
        **无归一化的问题**：
        - 第1层输出范围: [-10, 10]
        - 第10层输出范围: [-1000, 1000]（范围扩大）
        - 第20层输出范围: 可能溢出到inf
        
        **BatchNorm的作用**：
        - 强制每层输出均值=0，方差=1
        - 保持激活值在合理范围内
        - 梯度更稳定
        """
        )

    elif case_id == "huge_lr":
        st.subheader("🔥 学习率对比实验")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ❌ 学习率过大 (lr=10.0)")
            if st.button("模拟训练（大学习率）", key="train_huge_lr"):
                with st.spinner("训练中..."):
                    bad_losses = simulate_training_with_lr(
                        model,
                        case_info["input_size"],
                        case_info["bad_lr"],
                        num_steps=30,
                    )

                fig = plot_loss_curve(bad_losses, "Loss曲线（lr=10.0）")
                st.plotly_chart(fig, use_container_width=True)

                if any(np.isnan(bad_losses)):
                    st.error("💥 Loss变成NaN！梯度爆炸导致数值溢出")

        with col2:
            st.markdown("#### ✅ 合理学习率 (lr=0.01)")
            if st.button("模拟训练（正常学习率）", key="train_good_lr"):
                with st.spinner("训练中..."):
                    good_losses = simulate_training_with_lr(
                        model,
                        case_info["input_size"],
                        case_info["good_lr"],
                        num_steps=30,
                    )

                fig = plot_loss_curve(good_losses, "Loss曲线（lr=0.01）")
                st.plotly_chart(fig, use_container_width=True)

                st.success("✅ Loss正常下降，训练稳定")

    # 总结
    st.markdown("---")
    st.subheader("📚 学习要点")

    st.markdown(
        f"""
    **通过这个案例，你应该看到**：
    1. **具体的数值问题**：不是"可能会失败"，而是"梯度真的是1e-10"
    2. **问题的根源**：{case_info['reason']}
    3. **实际的解决方案**：{case_info['solution']}
    
    **记住**：神经网络的设计不是玄学，每个选择都有数学和工程上的理由！
    """
    )


if __name__ == "__main__":
    # 测试运行
    failure_museum_tab()
