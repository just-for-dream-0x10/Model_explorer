"""
优化器深度分析模块
Optimizer Deep Analysis Module

涵盖 SGD、Momentum、Nesterov、AdaGrad、RMSprop、Adam、AdamW
以及学习率调度策略的交互式可视化分析。
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.visualization.chart_utils import ChartBuilder
from simple_latex import display_latex


# ============================================================
# 工具函数：损失函数与梯度
# ============================================================

def rosenbrock(x, y, a=1.0, b=100.0):
    """Rosenbrock 函数: f(x,y) = (a-x)^2 + b*(y-x^2)^2"""
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(x, y, a=1.0, b=100.0):
    """Rosenbrock 函数的梯度"""
    dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    dy = 2 * b * (y - x ** 2)
    return np.array([dx, dy])


def beale(x, y):
    """Beale 函数"""
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2
    return term1 + term2 + term3


def beale_grad(x, y):
    """Beale 函数的梯度"""
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y ** 2
    term3 = 2.625 - x + x * y ** 3
    dx = 2 * term1 * (-1 + y) + 2 * term2 * (-1 + y ** 2) + 2 * term3 * (-1 + y ** 3)
    dy = (2 * term1 * x + 2 * term2 * 2 * x * y + 2 * term3 * 3 * x * y ** 2)
    return np.array([dx, dy])


# ============================================================
# 优化器实现（纯 numpy）
# ============================================================

def optimize_sgd(grad_fn, x0, lr, steps=200, clip=1e6):
    """SGD 优化器"""
    x = np.array(x0, dtype=np.float64)
    trajectory = [x.copy()]
    for _ in range(steps):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -clip, clip)
        x = x - lr * g
        trajectory.append(x.copy())
    return np.array(trajectory)


def optimize_momentum(grad_fn, x0, lr, gamma=0.9, steps=200, clip=1e6):
    """Momentum 优化器"""
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    for _ in range(steps):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -clip, clip)
        v = gamma * v + lr * g
        x = x - v
        trajectory.append(x.copy())
    return np.array(trajectory)


def optimize_nesterov(grad_fn, x0, lr, gamma=0.9, steps=200, clip=1e6):
    """Nesterov Accelerated Gradient"""
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    for _ in range(steps):
        x_ahead = x - gamma * v
        g = grad_fn(x_ahead[0], x_ahead[1])
        g = np.clip(g, -clip, clip)
        v = gamma * v + lr * g
        x = x - v
        trajectory.append(x.copy())
    return np.array(trajectory)


def optimize_adagrad(grad_fn, x0, lr, eps=1e-8, steps=200, clip=1e6):
    """AdaGrad 优化器"""
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    for _ in range(steps):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -clip, clip)
        v = v + g ** 2
        x = x - lr * g / (np.sqrt(v) + eps)
        trajectory.append(x.copy())
    return np.array(trajectory)


def optimize_rmsprop(grad_fn, x0, lr, gamma=0.9, eps=1e-8, steps=200, clip=1e6):
    """RMSprop 优化器"""
    x = np.array(x0, dtype=np.float64)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    for _ in range(steps):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -clip, clip)
        v = gamma * v + (1 - gamma) * g ** 2
        x = x - lr * g / (np.sqrt(v) + eps)
        trajectory.append(x.copy())
    return np.array(trajectory)


def optimize_adam(grad_fn, x0, lr, beta1=0.9, beta2=0.999, eps=1e-8,
                  steps=200, weight_decay=0.0, decoupled=False, clip=1e6):
    """Adam / AdamW 优化器"""
    x = np.array(x0, dtype=np.float64)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -clip, clip)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        if decoupled:
            x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)
        else:
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            if weight_decay > 0:
                x = x / (1 + lr * weight_decay)
        trajectory.append(x.copy())
    return np.array(trajectory)


# ============================================================
# 带缓存的计算函数
# ============================================================

@st.cache_data
def compute_contour_data(func_name, x_range, y_range, resolution):
    """计算等高线网格数据"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    if func_name == "Rosenbrock":
        Z = rosenbrock(X, Y)
    else:
        Z = beale(X, Y)
    return X, Y, Z


@st.cache_data
def compute_trajectory(_func_name, optimizer_name, x0, lr, momentum,
                       steps, clip_val):
    """计算优化轨迹（缓存）"""
    if _func_name == "Rosenbrock":
        grad_fn = rosenbrock_grad
    else:
        grad_fn = beale_grad

    if optimizer_name == "SGD":
        return optimize_sgd(grad_fn, x0, lr, steps, clip_val)
    elif optimizer_name == "Momentum":
        return optimize_momentum(grad_fn, x0, lr, momentum, steps, clip_val)
    elif optimizer_name == "Nesterov":
        return optimize_nesterov(grad_fn, x0, lr, momentum, steps, clip_val)
    elif optimizer_name == "AdaGrad":
        return optimize_adagrad(grad_fn, x0, lr, steps=steps, clip=clip_val)
    elif optimizer_name == "RMSprop":
        return optimize_rmsprop(grad_fn, x0, lr, momentum, steps=steps, clip=clip_val)
    elif optimizer_name == "Adam":
        return optimize_adam(grad_fn, x0, lr, steps=steps, clip=clip_val)
    elif optimizer_name == "AdamW":
        return optimize_adam(grad_fn, x0, lr, steps=steps, clip=clip_val)
    return np.array([x0])


@st.cache_data
def compute_adagrad_accumulation(grad_fn_name, steps, lr):
    """计算 AdaGrad 累积梯度平方"""
    np.random.seed(42)
    # 模拟一个有稀疏梯度的场景
    x = np.array([-1.5, 2.0], dtype=np.float64)
    if grad_fn_name == "Rosenbrock":
        grad_fn = rosenbrock_grad
    else:
        grad_fn = beale_grad

    v = np.zeros(2)
    history = {"step": [], "v_x": [], "v_y": [], "lr_x": [], "lr_y": []}
    for t in range(1, steps + 1):
        g = grad_fn(x[0], x[1])
        g = np.clip(g, -1e6, 1e6)
        v = v + g ** 2
        eff_lr = lr / (np.sqrt(v) + 1e-8)
        history["step"].append(t)
        history["v_x"].append(v[0])
        history["v_y"].append(v[1])
        history["lr_x"].append(eff_lr[0])
        history["lr_y"].append(eff_lr[1])
        x = x - eff_lr * g
    return history


@st.cache_data
def compute_lr_schedule(schedule_type, total_epochs, warmup_steps, min_lr,
                        max_lr, decay_rate):
    """计算学习率调度"""
    epochs = np.arange(1, total_epochs + 1)
    lr_list = []

    for e in epochs:
        if schedule_type == "Step Decay":
            # 每 total_epochs//3 步衰减一次
            period = max(1, total_epochs // 3)
            factor = decay_rate ** (e // period)
            lr_list.append(max_lr * factor)
        elif schedule_type == "Cosine Annealing":
            lr_list.append(
                min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * e / total_epochs))
            )
        elif schedule_type == "Warmup + Cosine":
            if e <= warmup_steps:
                lr_list.append(max_lr * e / warmup_steps)
            else:
                progress = (e - warmup_steps) / (total_epochs - warmup_steps)
                lr_list.append(
                    min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
                )
        elif schedule_type == "Exponential Decay":
            lr_list.append(max_lr * decay_rate ** e)

    return epochs.tolist(), lr_list


@st.cache_data
def compute_sensitivity_heatmap(optimizer_name, func_name, lr_values,
                                momentum_values, steps, x0):
    """计算超参数敏感性热力图"""
    if func_name == "Rosenbrock":
        grad_fn = rosenbrock_grad
        loss_fn = rosenbrock
    else:
        grad_fn = beale_grad
        loss_fn = beale

    results = np.zeros((len(momentum_values), len(lr_values)))
    for i, mom in enumerate(momentum_values):
        for j, lr_val in enumerate(lr_values):
            try:
                traj = compute_trajectory(
                    func_name, optimizer_name, x0, lr_val, mom, steps, 1e6
                )
                final_loss = loss_fn(traj[-1, 0], traj[-1, 1])
                results[i, j] = np.log10(final_loss + 1e-10)
            except Exception:
                results[i, j] = 10.0
    return results


@st.cache_data
def compute_adam_vs_adamw(weight_decay, steps, lr):
    """计算 Adam vs AdamW 参数幅度变化"""
    np.random.seed(42)
    x_adam = np.array([-1.5, 2.0], dtype=np.float64)
    x_adamw = np.array([-1.5, 2.0], dtype=np.float64)
    grad_fn = rosenbrock_grad

    m_a = np.zeros(2)
    v_a = np.zeros(2)
    m_w = np.zeros(2)
    v_w = np.zeros(2)

    history = {
        "step": [],
        "adam_norm": [],
        "adamw_norm": [],
        "adam_x": [],
        "adam_y": [],
        "adamw_x": [],
        "adamw_y": [],
    }

    for t in range(1, steps + 1):
        # Adam (L2 正则化)
        g_a = grad_fn(x_adam[0], x_adam[1])
        g_a = np.clip(g_a, -1e6, 1e6)
        m_a = 0.9 * m_a + 0.1 * g_a
        v_a = 0.999 * v_a + 0.001 * g_a ** 2
        m_hat = m_a / (1 - 0.9 ** t)
        v_hat = v_a / (1 - 0.999 ** t)
        x_adam = x_adam - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        x_adam = x_adam / (1 + lr * weight_decay)

        # AdamW (解耦权重衰减)
        g_w = grad_fn(x_adamw[0], x_adamw[1])
        g_w = np.clip(g_w, -1e6, 1e6)
        m_w = 0.9 * m_w + 0.1 * g_w
        v_w = 0.999 * v_w + 0.001 * g_w ** 2
        m_hat_w = m_w / (1 - 0.9 ** t)
        v_hat_w = v_w / (1 - 0.999 ** t)
        x_adamw = x_adamw - lr * (m_hat_w / (np.sqrt(v_hat_w) + 1e-8) + weight_decay * x_adamw)

        history["step"].append(t)
        history["adam_norm"].append(np.linalg.norm(x_adam))
        history["adamw_norm"].append(np.linalg.norm(x_adamw))
        history["adam_x"].append(x_adam[0])
        history["adam_y"].append(x_adam[1])
        history["adamw_x"].append(x_adamw[0])
        history["adamw_y"].append(x_adamw[1])

    return history


# ============================================================
# 辅助绘图函数
# ============================================================

def _build_contour_figure(chart, func_name, x_range, y_range, resolution=80):
    """构建等高线底图"""
    X, Y, Z = compute_contour_data(func_name, x_range, y_range, resolution)
    # 对 Z 取 log 以便更好可视化
    Z_log = np.log10(Z + 1e-10)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=Z_log, x=X[0, :], y=Y[:, 0],
        colorscale="Viridis",
        contours=dict(showlabels=True, labelfont=dict(size=8)),
        opacity=0.7,
        showscale=True,
        colorbar=dict(title="log₁₀(Loss)"),
    ))
    fig.update_layout(
        title=f"{func_name} Loss Landscape",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
    )
    return fig


def _add_trajectory_to_contour(fig, trajectory, name, color, show_markers=True):
    """在等高线图上添加优化轨迹"""
    fig.add_trace(go.Scatter(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        mode="lines+markers" if show_markers else "lines",
        name=name,
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
    ))
    # 标记起点和终点
    fig.add_trace(go.Scatter(
        x=[trajectory[0, 0]], y=[trajectory[0, 1]],
        mode="markers", name=f"{name} 起点",
        marker=dict(size=10, color=color, symbol="x", line_width=2),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[trajectory[-1, 0]], y=[trajectory[-1, 1]],
        mode="markers", name=f"{name} 终点",
        marker=dict(size=10, color=color, symbol="star", line_width=2),
        showlegend=False,
    ))


# ============================================================
# 主标签页函数
# ============================================================

def optimizer_analysis_tab(chinese_supported=True):
    """优化器深度分析标签页"""

    st.header("⚡ 优化器深度分析" if chinese_supported else "⚡ Optimizer Analysis")

    st.markdown(
        """
        深入理解神经网络训练中各类优化器的数学原理、行为特征与适用场景。
        通过交互式可视化对比 SGD、Momentum、AdaGrad、RMSprop、Adam、AdamW 等优化器。
        """
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 概览" if chinese_supported else "📊 Overview",
        "🚀 SGD & Momentum" if chinese_supported else "🚀 SGD & Momentum",
        "📈 自适应学习率" if chinese_supported else "📈 Adaptive LR",
        "⚖️ Adam vs AdamW" if chinese_supported else "⚖️ Adam vs AdamW",
        "📉 学习率调度" if chinese_supported else "📉 LR Schedule",
        "🌡️ 超参数敏感性" if chinese_supported else "🌡️ Sensitivity",
    ])

    chart = ChartBuilder()

    # ==========================================================
    # Section 1: 优化器概览
    # ==========================================================
    with tab1:
        st.subheader("优化器概览" if chinese_supported else "Optimizer Overview")

        st.markdown(
            """
            不同优化器在**收敛速度**、**内存开销**、**超参数敏感性**和**泛化能力**方面各有优劣。
            以下柱状图基于典型深度学习训练场景的参考数据（0-10 评分制，越高越好）。
            """
        )

        optimizers = ["SGD", "Momentum", "Nesterov", "AdaGrad", "RMSprop", "Adam", "AdamW"]

        # 参考数据（基于文献和实验经验）
        metrics = {
            "收敛速度": [3, 5, 6, 4, 6, 8, 8],
            "内存开销": [10, 9, 9, 8, 8, 7, 7],
            "超参数鲁棒性": [4, 5, 5, 3, 6, 7, 7],
            "泛化能力": [9, 8, 8, 5, 6, 6, 8],
        }

        metric_names_cn = list(metrics.keys())
        metric_names_en = ["Convergence Speed", "Memory Efficiency",
                           "Hyperparameter Robustness", "Generalization"]

        selected_metric = st.selectbox(
            "选择指标" if chinese_supported else "Select Metric",
            metric_names_cn,
            key="overview_metric",
        )

        metric_idx = metric_names_cn.index(selected_metric)
        values = metrics[selected_metric]

        fig = chart.create_bar_chart(
            x_data=optimizers,
            y_data=values,
            title=f"优化器对比 - {selected_metric}" if chinese_supported
            else f"Optimizer Comparison - {metric_names_en[metric_idx]}",
            x_title="优化器" if chinese_supported else "Optimizer",
            y_title="评分 (0-10)" if chinese_supported else "Score (0-10)",
            height=450,
        )
        chart.display_chart(fig)

        # 对比表格
        st.markdown("### 详细对比表" if chinese_supported else "### Detailed Comparison")
        import pandas as pd
        table_data = {
            "优化器": optimizers,
            "收敛速度": metrics["收敛速度"],
            "内存开销": metrics["内存开销"],
            "超参数鲁棒性": metrics["超参数鲁棒性"],
            "泛化能力": metrics["泛化能力"],
        }
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **关键洞察：**
            - **SGD** 泛化能力最强但收敛最慢，适合对精度要求极高的场景
            - **Adam / AdamW** 收敛最快，超参数鲁棒性好，是大多数场景的首选
            - **AdamW** 在 Adam 基础上改进了正则化方式，泛化能力显著提升
            - **AdaGrad** 的累积梯度平方会导致学习率单调递减，不适合深度网络
            """
        )

    # ==========================================================
    # Section 2: SGD & Momentum
    # ==========================================================
    with tab2:
        st.subheader("随机梯度下降与动量" if chinese_supported
                     else "SGD & Momentum")

        st.markdown("### 数学公式")

        st.markdown("**SGD (Stochastic Gradient Descent):**")
        st.latex(r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)")

        st.markdown("**Momentum:**")
        st.latex(r"v_t = \gamma \cdot v_{t-1} + \eta \nabla L(\theta_t)")
        st.latex(r"\theta_{t+1} = \theta_t - v_t")

        st.markdown("**Nesterov Accelerated Gradient (NAG):**")
        st.latex(r"v_t = \gamma \cdot v_{t-1} + \eta \nabla L(\theta_t - \gamma \cdot v_{t-1})")
        st.latex(r"\theta_{t+1} = \theta_t - v_t")

        st.markdown("其中 ")
        display_latex(r"\eta", display_mode=False)
        st.markdown(" 为学习率，")
        display_latex(r"\gamma", display_mode=False)
        st.markdown(" 为动量系数（通常取 0.9）。")

        st.markdown(
            """
            **Momentum 的直觉：** 想象一个球从山坡滚下，动量使其在一致的方向上加速，
            同时在梯度方向频繁变化时起到缓冲作用。Nesterov 则更进一步，
            先"向前看"一步再计算梯度，能更快响应方向变化。
            """
        )

        st.markdown("---")
        st.markdown("### 交互式 2D 优化轨迹")

        col1, col2, col3 = st.columns(3)
        with col1:
            func_name = st.selectbox(
                "损失函数" if chinese_supported else "Loss Function",
                ["Rosenbrock", "Beale"],
                key="sgd_func",
            )
        with col2:
            lr_sgd = st.slider(
                "学习率 (Learning Rate)",
                0.001, 1.0, 0.005, format="%.4f",
                key="sgd_lr",
            )
        with col3:
            momentum_val = st.slider(
                "动量 (Momentum γ)",
                0.0, 0.99, 0.9, format="%.2f",
                key="sgd_momentum",
            )

        steps_sgd = st.slider(
            "迭代步数" if chinese_supported else "Steps",
            50, 500, 200, key="sgd_steps",
        )

        x0_sgd = np.array([-1.5, 2.0])

        if func_name == "Rosenbrock":
            x_range, y_range = (-2.0, 2.0), (-1.0, 3.0)
        else:
            x_range, y_range = (-4.5, 4.5), (-4.5, 4.5)

        # 计算三条轨迹
        traj_sgd = compute_trajectory(
            func_name, "SGD", tuple(x0_sgd), lr_sgd, 0.0, steps_sgd, 1e6
        )
        traj_mom = compute_trajectory(
            func_name, "Momentum", tuple(x0_sgd), lr_sgd, momentum_val,
            steps_sgd, 1e6,
        )
        traj_nag = compute_trajectory(
            func_name, "Nesterov", tuple(x0_sgd), lr_sgd, momentum_val,
            steps_sgd, 1e6,
        )

        # 绘制三个子图
        fig_sgd = _build_contour_figure(chart, func_name, x_range, y_range)
        _add_trajectory_to_contour(fig_sgd, traj_sgd, "SGD", "#1f77b4")
        fig_sgd.update_layout(title="SGD 轨迹")

        fig_mom = _build_contour_figure(chart, func_name, x_range, y_range)
        _add_trajectory_to_contour(fig_mom, traj_mom, "Momentum", "#ff7f0e")
        fig_mom.update_layout(title="Momentum 轨迹")

        fig_nag = _build_contour_figure(chart, func_name, x_range, y_range)
        _add_trajectory_to_contour(fig_nag, traj_nag, "Nesterov", "#2ca02c")
        fig_nag.update_layout(title="Nesterov 轨迹")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            chart.display_chart(fig_sgd)
        with col_b:
            chart.display_chart(fig_mom)
        with col_c:
            chart.display_chart(fig_nag)

        # 叠加对比图
        fig_compare = _build_contour_figure(chart, func_name, x_range, y_range)
        _add_trajectory_to_contour(fig_compare, traj_sgd, "SGD", "#1f77b4")
        _add_trajectory_to_contour(fig_compare, traj_mom, "Momentum", "#ff7f0e")
        _add_trajectory_to_contour(fig_compare, traj_nag, "Nesterov", "#2ca02c")
        fig_compare.update_layout(title="三者叠加对比")
        chart.display_chart(fig_compare)

        # 最终 loss 对比
        if func_name == "Rosenbrock":
            loss_fn = rosenbrock
        else:
            loss_fn = beale

        loss_sgd = loss_fn(traj_sgd[-1, 0], traj_sgd[-1, 1])
        loss_mom = loss_fn(traj_mom[-1, 0], traj_mom[-1, 1])
        loss_nag = loss_fn(traj_nag[-1, 0], traj_nag[-1, 1])

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("SGD 最终 Loss", f"{loss_sgd:.6f}")
        with col_m2:
            st.metric("Momentum 最终 Loss", f"{loss_mom:.6f}")
        with col_m3:
            st.metric("Nesterov 最终 Loss", f"{loss_nag:.6f}")

    # ==========================================================
    # Section 3: 自适应学习率
    # ==========================================================
    with tab3:
        st.subheader("自适应学习率方法" if chinese_supported
                     else "Adaptive Learning Rate Methods")

        st.markdown("### 数学公式")

        st.markdown("**AdaGrad:**")
        st.latex(r"v_t = v_{t-1} + (\nabla L)^2")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \varepsilon} \nabla L")

        st.markdown("**RMSprop:**")
        st.latex(r"v_t = \gamma \cdot v_{t-1} + (1 - \gamma) (\nabla L)^2")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \varepsilon} \nabla L")

        st.markdown("**Adam (Adaptive Moment Estimation):**")
        st.latex(r"m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \nabla L")
        st.latex(r"v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) (\nabla L)^2")
        st.latex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}")

        st.markdown("其中 ")
        display_latex(r"\varepsilon", display_mode=False)
        st.markdown(" 通常取 ")
        display_latex(r"10^{-8}", display_mode=False)
        st.markdown("，防止除零。")

        st.markdown("---")

        # --- 交互 1: AdaGrad 累积 ---
        st.markdown("#### AdaGrad 梯度累积分析" if chinese_supported
                    else "#### AdaGrad Gradient Accumulation")

        st.markdown(
            """
            AdaGrad 的核心问题："""
        )
        display_latex(r"v_t", display_mode=False)
        st.markdown(
            """ 只增不减，导致有效学习率单调递减。
            对于频繁出现的特征，学习率快速下降；对于稀疏特征，学习率保持较大。
            """
        )

        lr_adagrad = st.slider(
            "AdaGrad 学习率",
            0.001, 0.1, 0.01, format="%.4f",
            key="adagrad_lr",
        )
        steps_adagrad = st.slider(
            "迭代步数",
            50, 300, 150, key="adagrad_steps",
        )

        adagrad_data = compute_adagrad_accumulation(
            "Rosenbrock", steps_adagrad, lr_adagrad
        )

        fig_accum = chart.create_line_chart(
            x_data=adagrad_data["step"],
            y_data=[
                adagrad_data["v_x"],
                adagrad_data["v_y"],
            ],
            title="AdaGrad 累积梯度平方 $v_t$",
            x_title="迭代步数" if chinese_supported else "Step",
            y_title="$v_t$",
            line_names=["$v_t^{(x)}$", "$v_t^{(y)}$"],
            height=400,
        )
        chart.display_chart(fig_accum)

        fig_eff_lr = chart.create_line_chart(
            x_data=adagrad_data["step"],
            y_data=[
                adagrad_data["lr_x"],
                adagrad_data["lr_y"],
            ],
            title="AdaGrad 有效学习率 $\\eta / (\\sqrt{v_t} + \\varepsilon)$",
            x_title="迭代步数" if chinese_supported else "Step",
            y_title="有效学习率" if chinese_supported else "Effective LR",
            line_names=["有效 LR (x)", "有效 LR (y)"],
            height=400,
        )
        chart.display_chart(fig_eff_lr)

        st.markdown("---")

        # --- 交互 2: AdaGrad vs RMSprop vs Adam 轨迹 ---
        st.markdown("#### 自适应优化器轨迹对比" if chinese_supported
                    else "#### Adaptive Optimizer Trajectory Comparison")

        col1, col2 = st.columns(2)
        with col1:
            func_name_ad = st.selectbox(
                "损失函数" if chinese_supported else "Loss Function",
                ["Rosenbrock", "Beale"],
                key="adaptive_func",
            )
        with col2:
            lr_adaptive = st.slider(
                "学习率",
                0.001, 0.1, 0.01, format="%.4f",
                key="adaptive_lr",
            )

        steps_adaptive = st.slider(
            "迭代步数",
            50, 500, 200, key="adaptive_steps",
        )

        x0_ad = np.array([-1.5, 2.0])

        if func_name_ad == "Rosenbrock":
            x_range_ad, y_range_ad = (-2.0, 2.0), (-1.0, 3.0)
        else:
            x_range_ad, y_range_ad = (-4.5, 4.5), (-4.5, 4.5)

        traj_adagrad = compute_trajectory(
            func_name_ad, "AdaGrad", tuple(x0_ad), lr_adaptive, 0.0,
            steps_adaptive, 1e6,
        )
        traj_rmsprop = compute_trajectory(
            func_name_ad, "RMSprop", tuple(x0_ad), lr_adaptive, 0.9,
            steps_adaptive, 1e6,
        )
        traj_adam = compute_trajectory(
            func_name_ad, "Adam", tuple(x0_ad), lr_adaptive, 0.0,
            steps_adaptive, 1e6,
        )

        fig_ad_compare = _build_contour_figure(
            chart, func_name_ad, x_range_ad, y_range_ad
        )
        _add_trajectory_to_contour(fig_ad_compare, traj_adagrad, "AdaGrad", "#d62728")
        _add_trajectory_to_contour(fig_ad_compare, traj_rmsprop, "RMSprop", "#ff7f0e")
        _add_trajectory_to_contour(fig_ad_compare, traj_adam, "Adam", "#2ca02c")
        fig_ad_compare.update_layout(title="AdaGrad vs RMSprop vs Adam")
        chart.display_chart(fig_ad_compare)

        if func_name_ad == "Rosenbrock":
            loss_fn_ad = rosenbrock
        else:
            loss_fn_ad = beale

        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            st.metric("AdaGrad 最终 Loss",
                      f"{loss_fn_ad(traj_adagrad[-1, 0], traj_adagrad[-1, 1]):.6f}")
        with col_l2:
            st.metric("RMSprop 最终 Loss",
                      f"{loss_fn_ad(traj_rmsprop[-1, 0], traj_rmsprop[-1, 1]):.6f}")
        with col_l3:
            st.metric("Adam 最终 Loss",
                      f"{loss_fn_ad(traj_adam[-1, 0], traj_adam[-1, 1]):.6f}")

    # ==========================================================
    # Section 4: Adam vs AdamW
    # ==========================================================
    with tab4:
        st.subheader("Adam vs AdamW" if chinese_supported else "Adam vs AdamW")

        st.markdown("### 数学公式")

        st.markdown("**Adam (L2 正则化):**")
        st.markdown("梯度更新与 L2 正则化耦合：")
        st.latex(r"\theta_{t+1} = \frac{\theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}}{1 + \eta \lambda}")

        st.markdown("**AdamW (解耦权重衰减):**")
        st.markdown("权重衰减直接作用于参数，与自适应学习率解耦：")
        st.latex(r"\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \cdot \theta_t \right)")

        st.markdown(
            """
            **核心区别：** Adam 中 L2 正则化被自适应学习率缩放，导致正则化效果不一致；
            AdamW 中权重衰减是独立的，不受梯度大小影响。
            """
        )

        st.markdown("---")
        st.markdown("#### 交互式对比" if chinese_supported
                    else "#### Interactive Comparison")

        wd_val = st.slider(
            "权重衰减 λ (Weight Decay)",
            0.0, 0.1, 0.01, format="%.4f",
            key="adamw_wd",
        )
        lr_aw = st.slider(
            "学习率",
            0.001, 0.05, 0.005, format="%.4f",
            key="adamw_lr",
        )
        steps_aw = st.slider(
            "迭代步数",
            100, 1000, 500, key="adamw_steps",
        )

        aw_data = compute_adam_vs_adamw(wd_val, steps_aw, lr_aw)

        # 参数范数变化
        fig_norm = chart.create_line_chart(
            x_data=aw_data["step"],
            y_data=[
                aw_data["adam_norm"],
                aw_data["adamw_norm"],
            ],
            title="参数范数变化 $\\|\\theta\\|$",
            x_title="迭代步数" if chinese_supported else "Step",
            y_title="$\\|\\theta\\|$",
            line_names=["Adam (L2)", "AdamW (Decoupled)"],
            height=400,
        )
        chart.display_chart(fig_norm)

        # 各参数分量变化
        fig_param = chart.create_line_chart(
            x_data=aw_data["step"],
            y_data=[
                aw_data["adam_x"],
                aw_data["adamw_x"],
            ],
            title="参数 x 分量变化",
            x_title="迭代步数" if chinese_supported else "Step",
            y_title="$\\theta_x$",
            line_names=["Adam $\\theta_x$", "AdamW $\\theta_x$"],
            height=400,
        )
        chart.display_chart(fig_param)

        fig_param2 = chart.create_line_chart(
            x_data=aw_data["step"],
            y_data=[
                aw_data["adam_y"],
                aw_data["adamw_y"],
            ],
            title="参数 y 分量变化",
            x_title="迭代步数" if chinese_supported else "Step",
            y_title="$\\theta_y$",
            line_names=["Adam $\\theta_y$", "AdamW $\\theta_y$"],
            height=400,
        )
        chart.display_chart(fig_param2)

        st.markdown(
            """
            **观察要点：**
            - 当 λ 较大时，Adam 的参数范数下降更快（因为 L2 与自适应缩放耦合）
            - AdamW 的权重衰减效果更稳定、更可预测
            - 在 Transformer 等大模型训练中，AdamW 已成为标准选择
            """
        )

    # ==========================================================
    # Section 5: 学习率调度
    # ==========================================================
    with tab5:
        st.subheader("学习率调度策略" if chinese_supported
                     else "Learning Rate Schedules")

        st.markdown("### 数学公式")

        st.markdown("**Step Decay (阶梯衰减):**")
        st.latex(r"\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}")

        st.markdown("**Cosine Annealing (余弦退火):**")
        st.latex(r"\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)")

        st.markdown("**Warmup + Cosine (预热 + 余弦):**")
        st.latex(r"\eta_t = \begin{cases} \eta_0 \cdot \frac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi (t - t_{\text{warmup}})}{T - t_{\text{warmup}}}\right)\right) & t > t_{\text{warmup}} \end{cases}")

        st.markdown("**Exponential Decay (指数衰减):**")
        st.latex(r"\eta_t = \eta_0 \cdot \gamma^t")

        st.markdown("---")
        st.markdown("#### 交互式调度对比" if chinese_supported
                    else "#### Interactive Schedule Comparison")

        col1, col2, col3 = st.columns(3)
        with col1:
            total_epochs = st.slider(
                "总 Epoch 数",
                50, 500, 200, key="sched_epochs",
            )
        with col2:
            warmup_steps = st.slider(
                "Warmup 步数",
                0, 100, 10, key="sched_warmup",
            )
        with col3:
            min_lr_val = st.slider(
                "最小学习率",
                0.0, 0.001, 0.0, format="%.5f",
                key="sched_min_lr",
            )

        max_lr_val = 0.001
        decay_rate = 0.1

        schedules = ["Step Decay", "Cosine Annealing", "Warmup + Cosine",
                     "Exponential Decay"]
        all_epochs = []
        all_lrs = []
        all_names = []

        colors_sched = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for sched in schedules:
            epochs, lrs = compute_lr_schedule(
                sched, total_epochs, warmup_steps, min_lr_val,
                max_lr_val, decay_rate,
            )
            all_epochs.append(epochs)
            all_lrs.append(lrs)
            all_names.append(sched)

        fig_sched = chart.create_line_chart(
            x_data=all_epochs[0],
            y_data=all_lrs,
            title="学习率调度策略对比",
            x_title="Epoch",
            y_title="Learning Rate",
            line_names=all_names,
            height=450,
        )
        chart.display_chart(fig_sched)

        st.markdown("---")
        st.markdown("#### Adam + Cosine Schedule 优化轨迹" if chinese_supported
                    else "#### Adam + Cosine Schedule Trajectory")

        st.markdown(
            """
            将 Cosine Annealing 学习率调度应用于 Adam 优化器，
            观察在训练后期学习率逐渐降低时优化轨迹的变化。
            """
        )

        lr_cosine = st.slider(
            "初始学习率",
            0.001, 0.05, 0.01, format="%.4f",
            key="cosine_lr",
        )
        cosine_steps = st.slider(
            "迭代步数",
            100, 500, 300, key="cosine_steps",
        )

        # Adam with cosine schedule
        x0_cos = np.array([-1.5, 2.0])
        grad_fn_cos = rosenbrock_grad

        x_cos = np.array(x0_cos, dtype=np.float64)
        m_cos = np.zeros(2)
        v_cos = np.zeros(2)
        traj_cosine = [x_cos.copy()]

        for t in range(1, cosine_steps + 1):
            # Cosine schedule
            progress = t / cosine_steps
            current_lr = min_lr_val + 0.5 * (lr_cosine - min_lr_val) * (
                1 + np.cos(np.pi * progress)
            )
            g = grad_fn_cos(x_cos[0], x_cos[1])
            g = np.clip(g, -1e6, 1e6)
            m_cos = 0.9 * m_cos + 0.1 * g
            v_cos = 0.999 * v_cos + 0.001 * g ** 2
            m_hat = m_cos / (1 - 0.9 ** t)
            v_hat = v_cos / (1 - 0.999 ** t)
            x_cos = x_cos - current_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            traj_cosine.append(x_cos.copy())

        traj_cosine = np.array(traj_cosine)

        # Adam without schedule for comparison
        traj_adam_fixed = compute_trajectory(
            "Rosenbrock", "Adam", tuple(x0_cos), lr_cosine, 0.0,
            cosine_steps, 1e6,
        )

        fig_cosine = _build_contour_figure(
            chart, "Rosenbrock", (-2.0, 2.0), (-1.0, 3.0)
        )
        _add_trajectory_to_contour(fig_cosine, traj_adam_fixed, "Adam (固定 LR)",
                                   "#1f77b4")
        _add_trajectory_to_contour(fig_cosine, traj_cosine, "Adam + Cosine",
                                   "#d62728")
        fig_cosine.update_layout(title="Adam + Cosine Schedule vs 固定学习率")
        chart.display_chart(fig_cosine)

        loss_fixed = rosenbrock(traj_adam_fixed[-1, 0], traj_adam_fixed[-1, 1])
        loss_cosine = rosenbrock(traj_cosine[-1, 0], traj_cosine[-1, 1])
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Adam (固定 LR) 最终 Loss", f"{loss_fixed:.6f}")
        with col_c2:
            st.metric("Adam + Cosine 最终 Loss", f"{loss_cosine:.6f}")

    # ==========================================================
    # Section 6: 超参数敏感性
    # ==========================================================
    with tab6:
        st.subheader("超参数敏感性分析" if chinese_supported
                     else "Hyperparameter Sensitivity Analysis")

        st.markdown(
            """
            不同优化器对超参数的敏感程度差异很大。
            以下热力图展示在 **学习率** 和 **动量/β₁** 的参数空间中，
            各优化器最终 Loss 的分布情况。
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            sens_optimizer = st.selectbox(
                "选择优化器" if chinese_supported else "Select Optimizer",
                ["SGD", "Momentum", "Adam", "RMSprop", "AdaGrad"],
                key="sens_opt",
            )
        with col2:
            sens_func = st.selectbox(
                "损失函数" if chinese_supported else "Loss Function",
                ["Rosenbrock", "Beale"],
                key="sens_func",
            )

        sens_steps = st.slider(
            "迭代步数",
            50, 300, 150, key="sens_steps",
        )

        # 学习率范围（对数刻度）
        lr_log_range = np.linspace(-4, -0.5, 20)
        lr_values = 10 ** lr_log_range

        # 动量/β1 范围
        if sens_optimizer in ["SGD", "AdaGrad"]:
            momentum_label = "未使用" if chinese_supported else "N/A"
            momentum_values = np.array([0.0])
        else:
            momentum_label = "动量/β₁"
            momentum_values = np.linspace(0.0, 0.99, 20)

        with st.spinner("计算超参数网格（可能需要几秒）..."):
            heatmap_data = compute_sensitivity_heatmap(
                sens_optimizer, sens_func, lr_values.tolist(),
                momentum_values.tolist(), sens_steps, (-1.5, 2.0),
            )

        # 绘制热力图
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"{lr:.1e}" for lr in lr_values],
            y=[f"{m:.2f}" for m in momentum_values],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="log₁₀(Loss)"),
        ))

        fig_heat.update_layout(
            title=f"{sens_optimizer} 超参数敏感性 ({sens_func})",
            xaxis_title="学习率 (Learning Rate, log scale)",
            yaxis_title=momentum_label,
            height=500,
            margin=dict(l=80, r=50, t=60, b=100),
            xaxis=dict(tickangle=45, dtick=3),
        )

        chart.display_chart(fig_heat)

        st.markdown(
            """
            **解读指南：**
            - **绿色区域**：Loss 较低，表示该超参数组合效果较好
            - **红色区域**：Loss 较高或发散，应避免这些超参数组合
            - **SGD** 对学习率非常敏感，好的区间很窄
            - **Adam** 对超参数更鲁棒，绿色区域更宽
            - **AdaGrad** 在学习率较大时容易发散
            """
        )

        # 额外：各优化器在固定动量下的学习率敏感性
        st.markdown("---")
        st.markdown("#### 学习率敏感性对比" if chinese_supported
                    else "#### Learning Rate Sensitivity Comparison")

        st.markdown("固定动量/β₁ = 0.9，对比各优化器在不同学习率下的表现。")

        fixed_momentum = 0.9
        optimizers_to_compare = ["SGD", "Momentum", "Adam", "RMSprop"]
        lr_compare = 10 ** np.linspace(-4, -0.5, 25)

        fig_lr_sens = go.Figure()
        opt_colors = {
            "SGD": "#1f77b4", "Momentum": "#ff7f0e",
            "Adam": "#2ca02c", "RMSprop": "#d62728",
        }

        for opt_name in optimizers_to_compare:
            losses = []
            for lr_val in lr_compare:
                try:
                    traj = compute_trajectory(
                        sens_func, opt_name, (-1.5, 2.0), lr_val,
                        fixed_momentum, sens_steps, 1e6,
                    )
                    if sens_func == "Rosenbrock":
                        final_loss = rosenbrock(traj[-1, 0], traj[-1, 1])
                    else:
                        final_loss = beale(traj[-1, 0], traj[-1, 1])
                    losses.append(np.log10(final_loss + 1e-10))
                except Exception:
                    losses.append(10.0)

            fig_lr_sens.add_trace(go.Scatter(
                x=[f"{lr:.1e}" for lr in lr_compare],
                y=losses,
                mode="lines+markers",
                name=opt_name,
                line=dict(color=opt_colors[opt_name], width=2),
                marker=dict(size=4),
            ))

        fig_lr_sens.update_layout(
            title=f"各优化器学习率敏感性 ({sens_func}, γ=0.9)",
            xaxis_title="学习率 (Learning Rate)",
            yaxis_title="log₁₀(Final Loss)",
            height=450,
            xaxis=dict(tickangle=45, dtick=4),
            margin=dict(l=50, r=50, t=60, b=100),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        chart.display_chart(fig_lr_sens)
