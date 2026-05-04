"""
损失函数深度解析模块
Loss Functions Deep Dive Module

涵盖回归损失、分类损失、对比损失、正则化项和损失景观的可视化与交互探索。
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.visualization.chart_utils import ChartBuilder
from simple_latex import display_latex


# ---------------------------------------------------------------------------
# 缓存计算函数
# ---------------------------------------------------------------------------

@st.cache_data
def _compute_regression_losses(errors, delta=1.0):
    """计算 MSE / MAE / Huber 损失值（缓存）"""
    mse = errors ** 2
    mae = np.abs(errors)
    huber = np.where(
        np.abs(errors) <= delta,
        0.5 * errors ** 2,
        delta * (np.abs(errors) - 0.5 * delta),
    )
    return mse, mae, huber


@st.cache_data
def _compute_regression_gradients(errors, delta=1.0):
    """计算 MSE / MAE / Huber 的梯度（对误差求导）"""
    grad_mse = 2.0 * errors
    grad_mae = np.sign(errors)
    grad_huber = np.where(
        np.abs(errors) <= delta,
        errors,
        delta * np.sign(errors),
    )
    return grad_mse, grad_mae, grad_huber


@st.cache_data
def _compute_focal_loss(p_t, gamma, alpha):
    """计算 Focal Loss（缓存）"""
    eps = 1e-8
    p_t_safe = np.clip(p_t, eps, 1.0 - eps)
    focal = -alpha * (1.0 - p_t_safe) ** gamma * np.log(p_t_safe)
    return focal


@st.cache_data
def _compute_label_smoothing(y_true, y_pred, epsilon, K):
    """计算 Label Smoothing 交叉熵（缓存）"""
    eps = 1e-8
    y_smooth = y_true * (1.0 - epsilon) + epsilon / K
    y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_smooth * np.log(y_pred_safe), axis=-1)
    return loss


@st.cache_data
def _compute_triplet_loss(d_ap, d_an, margin):
    """计算 Triplet Loss（缓存）"""
    return np.maximum(d_ap - d_an + margin, 0.0)


@st.cache_data
def _compute_infonce_loss(sim_pos, sim_negs, temperature):
    """计算 InfoNCE Loss（缓存）"""
    exp_pos = np.exp(sim_pos / temperature)
    exp_all = exp_pos + np.sum(np.exp(sim_negs / temperature))
    return -np.log(exp_pos / exp_all)


@st.cache_data
def _compute_regularization_weights(n_weights, l1_lambda, l2_lambda, seed=42):
    """模拟 L1/L2/Elastic Net 正则化后的权重分布（缓存）"""
    rng = np.random.RandomState(seed)
    # 模拟无正则化的原始权重
    raw = rng.randn(n_weights) * 2.0
    # L1 软阈值
    w_l1 = np.sign(raw) * np.maximum(np.abs(raw) - l1_lambda, 0.0)
    # L2 缩放
    w_l2 = raw * (1.0 / (1.0 + l2_lambda))
    # Elastic Net
    w_en = np.sign(raw) * np.maximum(np.abs(raw) - l1_lambda, 0.0) * (1.0 / (1.0 + l2_lambda))
    return raw, w_l1, w_l2, w_en


@st.cache_data
def _compute_loss_landscape_2d(func_name, resolution=80):
    """计算 2D 损失景观网格（缓存）"""
    x = np.linspace(-3, 3, resolution)
    y = np.linspace(-3, 3, resolution)
    X, Y = np.meshgrid(x, y)
    if func_name == "Rosenbrock":
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    elif func_name == "Himmelblau":
        Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    elif func_name == "Saddle":
        Z = X ** 2 - Y ** 2
    elif func_name == "Ackley":
        Z = (
            -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (X ** 2 + Y ** 2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)))
            + np.e + 20.0
        )
    else:
        Z = X ** 2 + Y ** 2
    return X, Y, Z


# ---------------------------------------------------------------------------
# Section 1: 概览
# ---------------------------------------------------------------------------

def _section_overview():
    """Section 1 - 损失函数概览"""
    st.header("1. 损失函数概览 Loss Functions Overview")

    st.markdown(
        """
        损失函数（Loss Function）是衡量模型预测值与真实值之间差异的函数，
        是训练神经网络时的优化目标。选择合适的损失函数对模型性能至关重要。
        """
    )

    overview_data = [
        {
            "类别": "回归损失 Regression",
            "损失函数": "MSE / MAE / Huber Loss",
            "公式": "$L = \\frac{1}{n}\\sum(y_i - \\hat{y}_i)^2$",
            "典型应用": "房价预测、温度回归等连续值任务",
        },
        {
            "类别": "分类损失 Classification",
            "损失函数": "Cross-Entropy / Focal Loss",
            "公式": "$L = -\\sum y_i \\log(\\hat{y}_i)$",
            "典型应用": "图像分类、文本分类等多分类任务",
        },
        {
            "类别": "对比损失 Contrastive",
            "损失函数": "Triplet Loss / InfoNCE",
            "公式": "$L = \\max(d(a,p) - d(a,n) + m, 0)$",
            "典型应用": "人脸识别、度量学习、自监督学习",
        },
        {
            "类别": "正则化 Regularization",
            "损失函数": "L1 / L2 / Elastic Net",
            "公式": "$L_{reg} = \\lambda \\sum |w_i| + \\lambda \\sum w_i^2$",
            "典型应用": "防止过拟合、特征选择",
        },
    ]

    st.table(overview_data)

    st.info(
        "**提示**: 下方各小节提供了每种损失函数的公式推导、交互式图表和参数调节功能。"
    )


# ---------------------------------------------------------------------------
# Section 2: 回归损失
# ---------------------------------------------------------------------------

def _section_regression_losses():
    """Section 2 - 回归损失"""
    st.header("2. 回归损失 Regression Losses")

    st.markdown(
        """
        回归损失用于衡量模型对连续值预测的偏差。下面介绍三种最常见的回归损失函数。
        """
    )

    # --- 公式展示 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("MSE 均方误差")
        display_latex(r"L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
        st.markdown("对大误差惩罚更重，梯度随误差线性增长。")
    with col2:
        st.subheader("MAE 平均绝对误差")
        display_latex(r"L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
        st.markdown("对异常值更鲁棒，梯度恒定。")
    with col3:
        st.subheader("Huber Loss")
        display_latex(
            r"L = \begin{cases}"
            r"\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\"
            r"\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}"
            r"\end{cases}"
        )
        st.markdown("MSE 与 MAE 的折中，通过 ")
        display_latex(r"\delta", display_mode=False)
        st.markdown(" 控制过渡。")

    # --- 交互：三种损失对比 ---
    st.markdown("---")
    st.subheader("交互：三种回归损失对比")

    delta = st.slider("Huber Loss 的 $\\delta$ 值", 0.1, 5.0, 1.0, 0.1, key="huber_delta")

    errors = np.linspace(-4, 4, 500)
    mse, mae, huber = _compute_regression_losses(errors, delta)

    chart = ChartBuilder()
    fig = chart.create_line_chart(
        x_data=errors.tolist(),
        y_data=[mse.tolist(), mae.tolist(), huber.tolist()],
        title="回归损失函数对比",
        x_title="误差 $e = y - \\hat{y}$",
        y_title="损失值",
        line_names=["MSE", "MAE", f"Huber ($\\delta$={delta})"],
        height=450,
    )
    chart.display_chart(fig)

    # --- 梯度图 ---
    st.markdown("---")
    st.subheader("交互：损失函数的梯度（导数）")

    grad_mse, grad_mae, grad_huber = _compute_regression_gradients(errors, delta)

    fig2 = chart.create_line_chart(
        x_data=errors.tolist(),
        y_data=[grad_mse.tolist(), grad_mae.tolist(), grad_huber.tolist()],
        title="回归损失函数的梯度",
        x_title="误差 $e$",
        y_title="$\\partial L / \\partial e$",
        line_names=["MSE 梯度", "MAE 梯度", f"Huber 梯度 ($\\delta$={delta})"],
        height=450,
    )
    chart.display_chart(fig2)

    st.markdown(
        """
        **观察要点：**
        - MSE 梯度与误差成正比，大误差时梯度非常大 → 容易受异常值影响
        - MAE 梯度恒为 ±1，对异常值鲁棒但在零点不可导
        - Huber 在误差 ≤ δ 时类似 MSE，超过 δ 后退化为线性 → 兼顾两者
        """
    )

    # --- 异常值影响 ---
    st.markdown("---")
    st.subheader("交互：异常值对损失的影响")

    n_normal = st.slider("正常样本数", 10, 100, 50, key="outlier_n_normal")
    outlier_val = st.slider("异常值大小", 2.0, 20.0, 10.0, 0.5, key="outlier_val")
    n_outliers = st.slider("异常值个数", 0, 20, 3, key="outlier_n_outliers")

    rng = np.random.RandomState(0)
    normal_errors = rng.randn(n_normal) * 0.5
    outlier_errors = np.full(n_outliers, outlier_val)
    all_errors = np.concatenate([normal_errors, outlier_errors])

    mse_val = np.mean(all_errors ** 2)
    mae_val = np.mean(np.abs(all_errors))
    huber_val = np.mean(np.where(
        np.abs(all_errors) <= delta,
        0.5 * all_errors ** 2,
        delta * (np.abs(all_errors) - 0.5 * delta),
    ))

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse_val:.4f}")
    col2.metric("MAE", f"{mae_val:.4f}")
    col3.metric("Huber Loss", f"{huber_val:.4f}")

    fig3 = chart.create_bar_chart(
        x_data=["MSE", "MAE", "Huber"],
        y_data=[mse_val, mae_val, huber_val],
        title="异常值下三种损失对比",
        x_title="损失函数",
        y_title="平均损失",
        height=400,
    )
    chart.display_chart(fig3)

    st.markdown(
        """
        **结论：** 当存在异常值时，MSE 的值会急剧增大，而 MAE 和 Huber 受影响较小。
        这就是为什么在含噪声的数据集上推荐使用 MAE 或 Huber Loss。
        """
    )


# ---------------------------------------------------------------------------
# Section 3: 分类损失
# ---------------------------------------------------------------------------

def _section_classification_losses():
    """Section 3 - 分类损失"""
    st.header("3. 分类损失 Classification Losses")

    st.markdown(
        """
        分类损失用于衡量模型对类别概率预测的准确性。交叉熵是最常用的分类损失，
        而 Focal Loss 和 Label Smoothing 是其重要的变体。
        """
    )

    # --- 公式 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Cross-Entropy 交叉熵")
        display_latex(r"L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)")
        st.markdown("多分类标准损失，配合 Softmax 使用。")
    with col2:
        st.subheader("Binary Cross-Entropy")
        display_latex(
            r"L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]"
        )
        st.markdown("二分类专用，配合 Sigmoid 使用。")
    with col3:
        st.subheader("Focal Loss")
        display_latex(
            r"L = -\alpha_t (1 - p_t)^\gamma \log(p_t)"
        )
        st.markdown("自动降低易分类样本的权重，聚焦难分类样本。")

    # --- BCE 曲线 ---
    st.markdown("---")
    st.subheader("Binary Cross-Entropy 可视化")

    p = np.linspace(0.01, 0.99, 200)
    bce_y1 = -np.log(p)         # y=1
    bce_y0 = -np.log(1.0 - p)   # y=0

    chart = ChartBuilder()
    fig = chart.create_line_chart(
        x_data=p.tolist(),
        y_data=[bce_y1.tolist(), bce_y0.tolist()],
        title="Binary Cross-Entropy Loss",
        x_title="预测概率 $\\hat{y}$",
        y_title="损失值",
        line_names=["$y=1$", "$y=0$"],
        height=400,
    )
    chart.display_chart(fig)

    # --- Focal Loss 交互 ---
    st.markdown("---")
    st.subheader("交互：Focal Loss 参数调节")

    col1, col2 = st.columns(2)
    with col1:
        gamma = st.slider("聚焦参数 $\\gamma$（focusing parameter）", 0.0, 5.0, 2.0, 0.1, key="focal_gamma")
    with col2:
        alpha = st.slider("平衡参数 $\\alpha$（balancing parameter）", 0.1, 1.0, 0.25, 0.05, key="focal_alpha")

    p_t = np.linspace(0.01, 0.99, 300)

    # 绘制不同 gamma 值的对比
    gamma_values = [0.0, 0.5, 1.0, 2.0, gamma]
    fig2 = go.Figure()
    colors = ChartBuilder.DEFAULT_COLORS
    for i, g in enumerate(gamma_values):
        fl = _compute_focal_loss(p_t, g, alpha)
        label = f"$\\gamma$={g:.1f}"
        fig2.add_trace(
            go.Scatter(
                x=p_t, y=fl, mode="lines", name=label,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig2.update_layout(
        title=dict(text=f"Focal Loss 曲线 ($\\alpha$={alpha})", x=0.5),
        xaxis_title="$p_t$（正确类别的预测概率）",
        yaxis_title="损失值",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig2)

    st.markdown(
        f"""
        **观察要点：**
        - $\\gamma = 0$ 时，Focal Loss 退化为标准交叉熵
        - $\\gamma$ 越大，易分类样本（$p_t$ 接近 1）的损失被压得越低
        - 当前 $\\gamma = {gamma:.1f}$ 时，当 $p_t = 0.9$ 的损失仅为交叉熵的
          ${(1.0 - 0.9) ** gamma:.4f} 倍
        """
    )

    # --- 调制因子 ---
    st.markdown("---")
    st.subheader("Focal Loss 调制因子 $(1 - p_t)^\\gamma$")

    fig3 = go.Figure()
    for i, g in enumerate(gamma_values):
        mod = (1.0 - p_t) ** g
        fig3.add_trace(
            go.Scatter(
                x=p_t, y=mod, mode="lines", name=f"$\\gamma$={g:.1f}",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig3.update_layout(
        title="调制因子 $(1 - p_t)^\\gamma$",
        xaxis_title="$p_t$",
        yaxis_title="调制因子",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig3)

    # --- Label Smoothing ---
    st.markdown("---")
    st.subheader("交互：Label Smoothing 标签平滑")

    epsilon = st.slider("平滑系数 $\\varepsilon$", 0.0, 0.5, 0.1, 0.01, key="label_smooth_eps")
    K = 5  # 类别数

    st.markdown("Label Smoothing 公式：")
    display_latex(r"L = -\sum_{i=1}^{K} \left(y_i(1-\varepsilon) + \frac{\varepsilon}{K}\right) \log(\hat{y}_i)")

    # 展示标签分布变化
    fig4 = go.Figure()
    classes = list(range(K))
    hard_label = [1.0 if i == 0 else 0.0 for i in classes]
    smooth_label = [(1.0 - epsilon) if i == 0 else epsilon / K for i in classes]

    fig4.add_trace(
        go.Bar(
            x=[f"类别 {c}" for c in classes], y=hard_label,
            name="Hard Label ($\\varepsilon=0$)", marker_color=colors[0],
        )
    )
    fig4.add_trace(
        go.Bar(
            x=[f"类别 {c}" for c in classes], y=smooth_label,
            name=f"Smooth Label ($\\varepsilon$={epsilon})", marker_color=colors[1],
        )
    )
    fig4.update_layout(
        title="标签平滑效果",
        xaxis_title="类别",
        yaxis_title="标签值",
        barmode="group",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig4)

    # 展示损失值变化
    p_correct = np.linspace(0.5, 1.0, 200)
    # 标准交叉熵（one-hot）
    ce_standard = -np.log(p_correct)
    # Label Smoothing 交叉熵
    ce_smooth = -(
        (1.0 - epsilon) * np.log(p_correct)
        + (K - 1) * (epsilon / K) * np.log((1.0 - p_correct) / (K - 1) + 1e-10)
    )

    fig5 = chart.create_line_chart(
        x_data=p_correct.tolist(),
        y_data=[ce_standard.tolist(), ce_smooth.tolist()],
        title="Label Smoothing 对损失值的影响",
        x_title="正确类别的预测概率",
        y_title="损失值",
        line_names=["标准 Cross-Entropy", f"Label Smoothing ($\\varepsilon$={epsilon})"],
        height=400,
    )
    chart.display_chart(fig5)

    st.markdown(
        """
        **Label Smoothing 的作用：**
        - 防止模型对预测过于自信（overconfidence）
        - 提高模型的泛化能力和校准（calibration）
        - 正则化效果，有助于训练更稳定的模型
        """
    )


# ---------------------------------------------------------------------------
# Section 4: 对比损失
# ---------------------------------------------------------------------------

def _section_contrastive_losses():
    """Section 4 - 对比损失"""
    st.header("4. 对比损失 Contrastive Losses")

    st.markdown(
        """
        对比损失通过拉近相似样本、推远不相似样本来学习良好的表示。
        广泛应用于度量学习、自监督学习和检索任务。
        """
    )

    # --- 公式 ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Triplet Loss 三元组损失")
        display_latex(
            r"L = \max(d(a, p) - d(a, n) + m, 0)"
        )
        st.markdown(
            """
            - **a**: Anchor（锚点样本）
            - **p**: Positive（正样本）
            - **n**: Negative（负样本）
            - **d**: 距离函数（通常为欧氏距离）
            - **m**: Margin（间隔）
            """
        )
    with col2:
        st.subheader("InfoNCE Loss")
        display_latex(
            r"L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}"
            r"{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k)/\tau)}"
        )
        st.markdown(
            """
            - **z_i, z_j**: 样本表示
            - **τ**: Temperature（温度参数）
            - **sim**: 余弦相似度
            - **N**: 批量大小
            """
        )

    # --- Triplet Loss 交互 ---
    st.markdown("---")
    st.subheader("交互：Triplet Loss 损失曲面")

    margin = st.slider("Margin $m$", 0.1, 3.0, 1.0, 0.1, key="triplet_margin")

    d_ap_range = np.linspace(0, 3, 100)
    d_an_range = np.linspace(0, 3, 100)
    D_ap, D_an = np.meshgrid(d_ap_range, d_an_range)
    L_triplet = np.maximum(D_ap - D_an + margin, 0.0)

    fig = go.Figure(data=go.Contour(
        z=L_triplet,
        x=d_ap_range,
        y=d_an_range,
        colorscale="Viridis",
        contours=dict(showlabels=True, labelfont=dict(size=10)),
    ))
    fig.update_layout(
        title=f"Triplet Loss 损失曲面 (margin={margin})",
        xaxis_title="$d(a, p)$ 正样本距离",
        yaxis_title="$d(a, n)$ 负样本距离",
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    chart = ChartBuilder()
    chart.display_chart(fig)

    st.markdown(
        f"""
        **解读：**
        - 当 $d(a,p) - d(a,n) + m \\le 0$（即 $d(a,n) \\ge d(a,p) + {margin}$）时，损失为 0
        - 等高线为 0 的区域是"安全区"，负样本距离足够远
        - Margin 越大，要求正负样本之间的间距越大
        """
    )

    # --- InfoNCE 交互 ---
    st.markdown("---")
    st.subheader("交互：InfoNCE 温度参数 $\\tau$ 的影响")

    temperature = st.slider(
        "温度 $\\tau$", 0.05, 1.0, 0.1, 0.05, key="infonce_tau"
    )
    n_negatives = st.slider("负样本数量", 1, 50, 10, key="infonce_n_neg")

    sim_pos_range = np.linspace(0.1, 1.0, 200)

    # 不同温度下的 InfoNCE 损失
    tau_values = [0.05, 0.1, 0.3, 0.5, temperature]
    fig2 = go.Figure()
    colors = ChartBuilder.DEFAULT_COLORS
    for i, tau in enumerate(tau_values):
        losses = []
        for sp in sim_pos_range:
            rng = np.random.RandomState(42)
            sim_negs = rng.uniform(-0.5, 0.8, n_negatives)
            loss = _compute_infonce_loss(sp, sim_negs, tau)
            losses.append(loss)
        fig2.add_trace(
            go.Scatter(
                x=sim_pos_range, y=losses, mode="lines",
                name=f"$\\tau$={tau:.2f}",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig2.update_layout(
        title="InfoNCE Loss vs 正样本相似度",
        xaxis_title="正样本相似度 $\\text{sim}(z_i, z_j)$",
        yaxis_title="InfoNCE Loss",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig2)

    st.markdown(
        f"""
        **温度参数 $\\tau$ 的作用：**
        - $\\tau$ 较小（如 0.05）：分布更尖锐，模型需要区分更细微的相似度差异
        - $\\tau$ 较大（如 0.5）：分布更平滑，对相似度差异的敏感度降低
        - 当前 $\\tau = {temperature:.2f}$，负样本数 = {n_negatives}
        """
    )

    # --- 相似度分布可视化 ---
    st.markdown("---")
    st.subheader("InfoNCE 相似度分布与 Softmax 概率")

    sim_values = np.array([0.8, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0, -0.1, -0.2, -0.3][:n_negatives + 1])
    sim_values = sim_values[:n_negatives + 1]

    tau_show = st.slider(
        "可视化温度 $\\tau$", 0.05, 1.0, 0.1, 0.05, key="infonce_tau_viz"
    )
    probs = np.exp(sim_values / tau_show)
    probs = probs / probs.sum()

    fig3 = go.Figure()
    bar_colors = [colors[0] if i == 0 else colors[4] for i in range(len(sim_values))]
    fig3.add_trace(
        go.Bar(
            x=[f"样本{i}" for i in range(len(sim_values))],
            y=probs,
            marker_color=bar_colors,
            text=[f"{p:.3f}" for p in probs],
            textposition="auto",
        )
    )
    fig3.update_layout(
        title=f"Softmax 概率分布 ($\\tau$={tau_show:.2f})",
        xaxis_title="样本（样本0为正样本）",
        yaxis_title="概率",
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig3)


# ---------------------------------------------------------------------------
# Section 5: 正则化
# ---------------------------------------------------------------------------

def _section_regularization():
    """Section 5 - 正则化项"""
    st.header("5. 正则化项 Regularization Terms")

    st.markdown(
        """
        正则化项被添加到损失函数中，用于约束模型参数，防止过拟合。
        不同的正则化方式会引导权重产生不同的分布特征。
        """
    )

    # --- 公式 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("L1 正则化")
        display_latex(r"L_{reg} = \lambda \sum_{i} |w_i|")
        st.markdown("产生稀疏权重，可用于特征选择。")
    with col2:
        st.subheader("L2 正则化")
        display_latex(r"L_{reg} = \lambda \sum_{i} w_i^2")
        st.markdown("权重衰减（Weight Decay），使权重趋向于较小的值。")
    with col3:
        st.subheader("Elastic Net")
        display_latex(r"L_{reg} = \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2")
        st.markdown("L1 与 L2 的结合，兼具稀疏性和稳定性。")

    # --- 权重分布对比 ---
    st.markdown("---")
    st.subheader("交互：L1 vs L2 权重分布")

    l1_lambda = st.slider("L1 系数 $\\lambda_1$", 0.0, 2.0, 0.5, 0.1, key="reg_l1")
    l2_lambda = st.slider("L2 系数 $\\lambda_2$", 0.0, 2.0, 0.5, 0.1, key="reg_l2")
    n_weights = st.slider("权重数量", 100, 2000, 500, 100, key="reg_n_weights")

    raw, w_l1, w_l2, w_en = _compute_regularization_weights(n_weights, l1_lambda, l2_lambda)

    chart = ChartBuilder()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=raw, nbinsx=50, name="原始权重", opacity=0.5, marker_color=colors[0]))
    fig.add_trace(go.Histogram(x=w_l1, nbinsx=50, name=f"L1 ($\\lambda$={l1_lambda})", opacity=0.5, marker_color=colors[1]))
    fig.add_trace(go.Histogram(x=w_l2, nbinsx=50, name=f"L2 ($\\lambda$={l2_lambda})", opacity=0.5, marker_color=colors[2]))

    fig.update_layout(
        title="权重分布对比：原始 vs L1 vs L2",
        xaxis_title="权重值",
        yaxis_title="频数",
        barmode="overlay",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig)

    # 统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("L1 零权重比例", f"{np.mean(np.abs(w_l1) < 1e-6):.1%}")
    with col2:
        st.metric("L2 零权重比例", f"{np.mean(np.abs(w_l2) < 1e-6):.1%}")
    with col3:
        st.metric("L1 权重 L0 范数", f"{np.sum(np.abs(w_l1) > 1e-6)}")

    # --- Elastic Net ---
    st.markdown("---")
    st.subheader("交互：Elastic Net 正则化")

    en_l1 = st.slider("Elastic Net $\\lambda_1$", 0.0, 2.0, 0.3, 0.1, key="en_l1")
    en_l2 = st.slider("Elastic Net $\\lambda_2$", 0.0, 2.0, 0.3, 0.1, key="en_l2")

    _, _, _, w_en = _compute_regularization_weights(n_weights, en_l1, en_l2)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=raw, nbinsx=50, name="原始权重", opacity=0.4, marker_color=colors[0]))
    fig2.add_trace(
        go.Histogram(x=w_en, nbinsx=50, name=f"Elastic Net ($\\lambda_1$={en_l1}, $\\lambda_2$={en_l2})",
                     opacity=0.6, marker_color=colors[3])
    )
    fig2.update_layout(
        title="Elastic Net 权重分布",
        xaxis_title="权重值",
        yaxis_title="频数",
        barmode="overlay",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart.display_chart(fig2)

    # 正则化路径
    st.markdown("---")
    st.subheader("正则化路径 Regularization Path")

    lambda_range = np.linspace(0, 3.0, 50)
    l1_counts = []
    l2_maxes = []
    en_counts = []
    for lam in lambda_range:
        _, w1, w2, we = _compute_regularization_weights(n_weights, lam, lam)
        l1_counts.append(np.sum(np.abs(w1) > 1e-6))
        l2_maxes.append(np.max(np.abs(w2)))
        en_counts.append(np.sum(np.abs(we) > 1e-6))

    fig3 = chart.create_line_chart(
        x_data=lambda_range.tolist(),
        y_data=[l1_counts, l2_maxes, en_counts],
        title="正则化强度 vs 权重统计",
        x_title="正则化系数 $\\lambda$",
        y_title="统计量",
        line_names=["L1 非零权重数", "L2 最大权重绝对值", "Elastic Net 非零权重数"],
        height=400,
    )
    chart.display_chart(fig3)

    st.markdown(
        """
        **关键洞察：**
        - L1 正则化使权重趋向于精确零（稀疏性），适合特征选择
        - L2 正则化使权重趋向于较小但非零的值，适合防止过拟合
        - Elastic Net 结合两者优势，在高维相关特征场景下表现更好
        """
    )


# ---------------------------------------------------------------------------
# Section 6: 损失景观
# ---------------------------------------------------------------------------

def _section_loss_landscape():
    """Section 6 - 损失景观"""
    st.header("6. 损失景观 Loss Landscape")

    st.markdown(
        """
        损失景观描述了损失函数在参数空间中的几何形状。
        理解损失景观有助于理解优化算法的行为和训练动态。
        """
    )

    # --- 1D 损失曲线 ---
    st.subheader("1D 损失曲线")

    x_1d = np.linspace(-3, 3, 500)

    fig_1d = go.Figure()
    # 几种常见的 1D 损失景观
    y_quadratic = x_1d ** 2
    y_nonconvex = x_1d ** 4 - 3 * x_1d ** 2 + x_1d
    y_oscillatory = x_1d ** 2 + 0.5 * np.sin(5 * x_1d)

    fig_1d.add_trace(go.Scatter(x=x_1d, y=y_quadratic, mode="lines", name="凸函数 $x^2$"))
    fig_1d.add_trace(go.Scatter(x=x_1d, y=y_nonconvex, mode="lines", name="非凸函数 $x^4 - 3x^2 + x$"))
    fig_1d.add_trace(go.Scatter(x=x_1d, y=y_oscillatory, mode="lines", name="振荡函数 $x^2 + 0.5\\sin(5x)$"))

    fig_1d.update_layout(
        title="1D 损失景观示例",
        xaxis_title="参数 $\\theta$",
        yaxis_title="损失 $L(\\theta)$",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    chart = ChartBuilder()
    chart.display_chart(fig_1d)

    # --- 2D 损失景观 ---
    st.markdown("---")
    st.subheader("2D 损失景观（等高线图）")

    func_options = {
        "二次函数 Quadratic ($x^2 + y^2$)": "Quadratic",
        "Rosenbrock 函数": "Rosenbrock",
        "Himmelblau 函数": "Himmelblau",
        "鞍点 Saddle ($x^2 - y^2$)": "Saddle",
        "Ackley 函数": "Ackley",
    }
    func_names = list(func_options.keys())
    selected_func = st.selectbox("选择损失函数", func_names, key="landscape_func")
    func_key = func_options[selected_func]

    X, Y, Z = _compute_loss_landscape_2d(func_key)

    fig_2d = go.Figure(data=go.Contour(
        z=Z, x=X[0], y=Y[:, 0],
        colorscale="Viridis",
        contours=dict(showlabels=True, labelfont=dict(size=9)),
        ncontours=30,
    ))
    fig_2d.update_layout(
        title=f"2D 损失景观: {selected_func}",
        xaxis_title="$\\theta_1$",
        yaxis_title="$\\theta_2$",
        height=550,
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    chart.display_chart(fig_2d)

    # --- 3D 曲面图 ---
    st.markdown("---")
    st.subheader("3D 损失景观（曲面图）")

    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X[0], y=Y[:, 0], colorscale="Viridis",
                                        lighting=dict(ambient=0.6), lightposition=dict(x=0, y=0, z=1e5))])
    fig_3d.update_layout(
        title=f"3D 损失景观: {selected_func}",
        scene=dict(xaxis_title="$\\theta_1$", yaxis_title="$\\theta_2$", zaxis_title="$L$"),
        height=550,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    chart.display_chart(fig_3d)

    # --- 景观特征说明 ---
    st.markdown("---")
    st.subheader("损失景观特征说明")

    landscape_info = {
        "二次函数 Quadratic": {
            "特征": "全局唯一最小值，凸函数",
            "优化难度": "简单 - 梯度下降可直接收敛",
            "实际意义": "线性回归等简单模型的损失景观近似于此",
        },
        "Rosenbrock 函数": {
            "特征": "狭长的\"香蕉形\"山谷，全局最小值在 (1,1)",
            "优化难度": "困难 - 梯度在山谷中振荡",
            "实际意义": "模拟条件数差的优化问题",
        },
        "Himmelblau 函数": {
            "特征": "4 个等价的全局最小值",
            "优化难度": "中等 - 多个最优解",
            "实际意义": "模拟对称性导致的多个最优解",
        },
        "鞍点 Saddle": {
            "特征": "一个方向是极小值，另一个方向是极大值",
            "优化难度": "困难 - 梯度接近零但不是最小值",
            "实际意义": "高维空间中鞍点比局部最小值更常见",
        },
        "Ackley 函数": {
            "特征": "大量局部最小值，全局最小值在原点",
            "优化难度": "非常困难 - 容易陷入局部最小值",
            "实际意义": "模拟高度非凸的神经网络损失景观",
        },
    }

    info = landscape_info.get(selected_func, {})
    if info:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**特征**: {info['特征']}")
        col2.markdown(f"**优化难度**: {info['优化难度']}")
        col3.markdown(f"**实际意义**: {info['实际意义']}")

    st.markdown(
        """
        **关于损失景观的重要知识：**
        - 深度神经网络的损失景观通常高度非凸，存在大量鞍点和局部最小值
        - 研究表明，大多数局部最小值的损失值接近全局最小值（不会太差）
        - SGD 的噪声有助于逃离鞍点和浅的局部最小值
        - Batch Normalization、残差连接等技术可以改善损失景观的平滑度
        """
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def loss_functions_tab(chinese_supported=True):
    """损失函数深度解析标签页

    Args:
        chinese_supported: 是否支持中文显示
    """
    st.title("损失函数深度解析 Loss Functions Explorer")

    st.markdown(
        """
        本模块提供神经网络中常用损失函数的全面解析，包括公式推导、交互式可视化
        和参数调节功能。通过直观的图表和交互控件，帮助理解各种损失函数的特性与适用场景。
        """
    )

    # 使用 tabs 组织六大板块
    tab_labels = [
        "概览 Overview",
        "回归损失 Regression",
        "分类损失 Classification",
        "对比损失 Contrastive",
        "正则化 Regularization",
        "损失景观 Landscape",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _section_overview()

    with tabs[1]:
        _section_regression_losses()

    with tabs[2]:
        _section_classification_losses()

    with tabs[3]:
        _section_contrastive_losses()

    with tabs[4]:
        _section_regularization()

    with tabs[5]:
        _section_loss_landscape()
