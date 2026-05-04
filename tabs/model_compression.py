"""
模型压缩分析标签页
Model Compression Analysis Tab

综合分析模型压缩技术：量化、知识蒸馏、结构化剪枝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from utils.visualization.chart_utils import ChartBuilder


# ============================================================
# 参考数据（基于已发表论文的典型结果）
# ============================================================
# 压缩技术对比数据
COMPRESSION_OVERVIEW = {
    "Pruning": {
        "compression_ratio": 3.0,       # 典型压缩比
        "accuracy_retention": 0.985,     # 精度保留率
        "speedup": 2.0,                  # 推理加速比
        "hardware_requirement": 2,       # 1=低 2=中 3=高
    },
    "Quantization": {
        "compression_ratio": 4.0,
        "accuracy_retention": 0.990,
        "speedup": 3.0,
        "hardware_requirement": 3,
    },
    "Distillation": {
        "compression_ratio": 2.5,
        "accuracy_retention": 0.975,
        "speedup": 2.5,
        "hardware_requirement": 1,
    },
    "Low-Rank Factorization": {
        "compression_ratio": 2.0,
        "accuracy_retention": 0.970,
        "speedup": 1.5,
        "hardware_requirement": 1,
    },
}

# 量化位宽参考数据（以 ResNet-50 为例）
QUANTIZATION_DATA = {
    "bit_widths": [32, 16, 8, 4],
    "accuracy": [76.15, 76.10, 75.80, 72.50],       # Top-1 Accuracy (%)
    "model_size_mb": [97.8, 48.9, 24.5, 12.3],       # 模型大小 (MB)
    "inference_speed_ms": [6.2, 3.8, 2.1, 1.4],      # 推理延迟 (ms)
}

# 不同模型的量化数据
MODEL_QUANT_DATA = {
    "ResNet-50": {
        "params_m": 25.6,
        "fp32_size_mb": 97.8,
        "fp32_acc": 76.15,
    },
    "MobileNetV2": {
        "params_m": 3.4,
        "fp32_size_mb": 13.6,
        "fp32_acc": 72.00,
    },
    "BERT-Base": {
        "params_m": 110.0,
        "fp32_size_mb": 418.0,
        "fp32_acc": 80.50,
    },
    "GPT-2 Small": {
        "params_m": 124.4,
        "fp32_size_mb": 475.0,
        "fp32_acc": 0.0,  # 语言模型，无分类精度
    },
    "ViT-Base": {
        "params_m": 86.6,
        "fp32_size_mb": 330.0,
        "fp32_acc": 81.80,
    },
}

# 剪枝参考数据（ResNet-50 on ImageNet）
PRUNING_DATA = {
    "ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "magnitude_acc": [76.15, 76.10, 75.90, 75.50, 74.80, 73.50, 71.20, 67.50, 60.00, 45.00],
    "gradient_acc": [76.15, 76.12, 76.00, 75.70, 75.10, 74.00, 72.00, 68.50, 62.00, 48.00],
    "flops_reduction": [0.0, 0.10, 0.19, 0.28, 0.36, 0.44, 0.51, 0.58, 0.64, 0.70],
}


# ============================================================
# 缓存计算函数
# ============================================================

@st.cache_data
def compute_quantization_impact(
    model_name: str, bit_width: int
) -> Dict[str, float]:
    """计算量化后的模型大小和精度变化

    Args:
        model_name: 模型名称
        bit_width: 目标位宽

    Returns:
        包含 size_mb, accuracy, size_ratio 的字典
    """
    info = MODEL_QUANT_DATA[model_name]
    fp32_size = info["fp32_size_mb"]
    fp32_acc = info["fp32_acc"]

    # 模型大小 = 参数量 * 位宽 / 8 (字节) / 1024^2 (MB)
    params = info["params_m"] * 1e6
    new_size_mb = params * bit_width / 8 / (1024 ** 2)
    size_ratio = new_size_mb / fp32_size

    # 精度衰减模型（经验公式）
    if fp32_acc > 0:
        # INT8 几乎无损，INT4 有明显衰减
        if bit_width >= 16:
            acc = fp32_acc
        elif bit_width >= 8:
            acc = fp32_acc - 0.35
        else:
            acc = fp32_acc - 3.65
    else:
        acc = 0.0

    return {
        "size_mb": round(new_size_mb, 1),
        "accuracy": round(acc, 2),
        "size_ratio": round(size_ratio, 3),
    }


@st.cache_data
def compute_temperature_distribution(
    logits: Tuple[float, ...], temperature: float
) -> Tuple[List[float], List[float]]:
    """计算不同温度下的 softmax 分布

    Args:
        logits: 原始 logits
        temperature: 温度参数

    Returns:
        (原始分布, 软化分布)
    """
    logits_arr = np.array(logits, dtype=np.float64)

    # 原始 softmax (T=1)
    exp_orig = np.exp(logits_arr - np.max(logits_arr))
    soft_orig = exp_orig / exp_orig.sum()

    # 温度缩放 softmax
    scaled = logits_arr / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    soft_scaled = exp_scaled / exp_scaled.sum()

    return soft_orig.tolist(), soft_scaled.tolist()


@st.cache_data
def compute_distillation_loss(
    alpha: float, temperature: float,
    hard_loss: float, soft_loss: float
) -> Tuple[float, float, float]:
    """计算知识蒸馏总损失

    L = alpha * L_hard + (1 - alpha) * L_soft * T^2

    Args:
        alpha: 硬标签损失权重
        temperature: 温度
        hard_loss: 硬标签交叉熵
        soft_loss: 软标签KL散度

    Returns:
        (总损失, 硬标签贡献, 软标签贡献)
    """
    hard_contrib = alpha * hard_loss
    soft_contrib = (1 - alpha) * soft_loss * (temperature ** 2)
    total = hard_contrib + soft_contrib
    return round(total, 4), round(hard_contrib, 4), round(soft_contrib, 4)


@st.cache_data
def compute_pruning_estimate(
    pruning_ratio: float, method: str = "magnitude"
) -> Dict[str, float]:
    """估算剪枝后的精度和模型大小

    Args:
        pruning_ratio: 剪枝比例 (0~1)
        method: 'magnitude' 或 'gradient'

    Returns:
        包含 accuracy, flops_reduction, size_ratio 的字典
    """
    ratios = np.array(PRUNING_DATA["ratios"])
    acc_data = PRUNING_DATA[f"{method}_acc"]
    acc_arr = np.array(acc_data)
    flops_arr = np.array(PRUNING_DATA["flops_reduction"])

    # 线性插值
    accuracy = float(np.interp(pruning_ratio, ratios, acc_arr))
    flops_reduction = float(np.interp(pruning_ratio, ratios, flops_arr))
    size_ratio = 1.0 - pruning_ratio

    return {
        "accuracy": round(accuracy, 2),
        "flops_reduction": round(flops_reduction, 3),
        "size_ratio": round(size_ratio, 3),
    }


@st.cache_data
def compute_pipeline_compression(
    pruning_ratio: float,
    quantization_bit: int,
    distillation_alpha: float,
) -> Dict[str, float]:
    """计算组合压缩流水线的累积效果

    流程: Pruning -> Quantization -> Distillation

    Args:
        pruning_ratio: 剪枝比例
        quantization_bit: 量化位宽
        distillation_alpha: 蒸馏中的 alpha

    Returns:
        累积压缩比和精度估算
    """
    # Step 1: Pruning
    prune_result = compute_pruning_estimate(pruning_ratio, "magnitude")
    prune_acc = prune_result["accuracy"]
    prune_size_ratio = prune_result["size_ratio"]

    # Step 2: Quantization (在剪枝后的模型上)
    # 量化额外压缩 = 32 / bit_width
    quant_compression = 32.0 / quantization_bit
    cumulative_compression = (1.0 / prune_size_ratio) * quant_compression

    # 量化精度衰减
    if quantization_bit >= 16:
        quant_acc_drop = 0.0
    elif quantization_bit >= 8:
        quant_acc_drop = 0.5
    else:
        quant_acc_drop = 3.0
    post_quant_acc = prune_acc - quant_acc_drop

    # Step 3: Distillation 恢复部分精度
    # 蒸馏通常能恢复 0.5~2% 的精度损失
    distillation_recovery = 0.5 + distillation_alpha * 1.5
    final_acc = post_quant_acc + distillation_recovery

    return {
        "pruning_acc": round(prune_acc, 2),
        "post_quant_acc": round(post_quant_acc, 2),
        "final_acc": round(final_acc, 2),
        "pruning_compression": round(1.0 / prune_size_ratio, 2),
        "quant_compression": round(quant_compression, 2),
        "cumulative_compression": round(cumulative_compression, 2),
        "distillation_recovery": round(distillation_recovery, 2),
    }


# ============================================================
# Section 1: 概览
# ============================================================

def _section_overview(chart: ChartBuilder):
    """模型压缩概览"""
    st.header("模型压缩概览")
    st.markdown(
        """
        模型压缩旨在在尽量保持精度的前提下，减小模型大小、加速推理。
        主要技术包括：**Pruning（剪枝）**、**Quantization（量化）**、
        **Knowledge Distillation（知识蒸馏）** 和 **Low-Rank Factorization（低秩分解）**。
        """
    )

    techniques = list(COMPRESSION_OVERVIEW.keys())
    compression_ratios = [COMPRESSION_OVERVIEW[t]["compression_ratio"] for t in techniques]
    accuracy_retentions = [COMPRESSION_OVERVIEW[t]["accuracy_retention"] * 100 for t in techniques]
    speedups = [COMPRESSION_OVERVIEW[t]["speedup"] for t in techniques]
    hw_reqs = [COMPRESSION_OVERVIEW[t]["hardware_requirement"] for t in techniques]

    # 使用子图同时展示多个指标
    fig = chart.create_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "压缩比 Compression Ratio",
            "精度保留率 Accuracy Retention (%)",
            "推理加速比 Speedup",
            "硬件需求 Hardware Requirement",
        ],
        height=600,
    )

    fig.add_trace(
        go.Bar(x=techniques, y=compression_ratios, marker_color=chart.colors[0],
               text=[f"{v:.1f}x" for v in compression_ratios], textposition="auto"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=techniques, y=accuracy_retentions, marker_color=chart.colors[1],
               text=[f"{v:.1f}%" for v in accuracy_retentions], textposition="auto"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=techniques, y=speedups, marker_color=chart.colors[2],
               text=[f"{v:.1f}x" for v in speedups], textposition="auto"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=techniques, y=hw_reqs, marker_color=chart.colors[3],
               text=["低" if v == 1 else ("中" if v == 2 else "高") for v in hw_reqs],
               textposition="auto"),
        row=2, col=2,
    )

    fig.update_layout(height=650)
    chart.display_chart(fig)

    # 对比表格
    st.markdown("#### 技术对比总结")
    comparison_md = """
| 技术 | 压缩比 | 精度保留 | 加速比 | 硬件需求 | 适用场景 |
|:-----|:------:|:-------:|:-----:|:-------:|:---------|
| Pruning | ~3x | ~98.5% | ~2x | 中 | 通用，需微调 |
| Quantization | ~4x | ~99% | ~3x | 高（INT8加速） | 边缘部署 |
| Distillation | ~2.5x | ~97.5% | ~2.5x | 低 | 模型迁移 |
| Low-Rank | ~2x | ~97% | ~1.5x | 低 | 全连接层压缩 |
"""
    st.markdown(comparison_md)


# ============================================================
# Section 2: 量化分析
# ============================================================

def _section_quantization(chart: ChartBuilder):
    """量化分析"""
    st.header("量化分析 Quantization Analysis")
    st.markdown(
        """
        量化将模型参数从高精度浮点数映射到低精度表示，从而减小模型体积并加速推理。
        常见路径：**FP32 -> FP16 -> INT8 -> INT4**
        """
    )

    # ---- 数学公式 ----
    st.markdown("#### 量化公式")
    st.markdown(
        r"""
        量化核心公式：

        $$q = \text{round}\left(\frac{r}{s}\right) + z$$

        其中：
        - $r$：原始浮点数值（real value）
        - $s$：缩放因子（scale factor），$s = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$
        - $z$：零点偏移（zero point），$z = \text{round}\left(\frac{-r_{\min}}{s}\right)$
        - $q$：量化后的整数值

        反量化：$r' = s \cdot (q - z)$
        """
    )

    # ---- 对称 vs 非对称量化 ----
    st.markdown("#### 对称量化 vs 非对称量化")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **对称量化 (Symmetric)**
            - 零点 $z = 0$
            - $s = \frac{\max(|r|)}{2^{b-1} - 1}$
            - 计算简单，硬件友好
            - 适合权重（通常对称分布）
            """
        )
    with col2:
        st.markdown(
            """
            **非对称量化 (Asymmetric)**
            - 零点 $z \neq 0$
            - $s = \frac{r_{\max} - r_{\min}}{2^b - 1}$
            - 更充分利用量化范围
            - 适合激活值（可能偏移分布）
            """
        )

    # ---- 位宽 vs 精度折线图 ----
    st.markdown("#### 位宽 vs 精度 / 模型大小 / 推理速度")
    bit_widths = QUANTIZATION_DATA["bit_widths"]
    bit_labels = [f"FP{b}" if b >= 16 else f"INT{b}" for b in bit_widths]

    fig_acc = chart.create_line_chart(
        x_data=bit_labels,
        y_data=[QUANTIZATION_DATA["accuracy"]],
        title="位宽 vs Top-1 精度 (ResNet-50, ImageNet)",
        x_title="数据类型",
        y_title="Top-1 Accuracy (%)",
        line_names=["精度"],
        height=350,
    )
    chart.display_chart(fig_acc)

    col1, col2 = st.columns(2)
    with col1:
        fig_size = chart.create_bar_chart(
            x_data=bit_labels,
            y_data=QUANTIZATION_DATA["model_size_mb"],
            title="位宽 vs 模型大小",
            x_title="数据类型",
            y_title="模型大小 (MB)",
            color=chart.colors[0],
            height=350,
        )
        chart.display_chart(fig_size)

    with col2:
        fig_speed = chart.create_bar_chart(
            x_data=bit_labels,
            y_data=QUANTIZATION_DATA["inference_speed_ms"],
            title="位宽 vs 推理延迟",
            x_title="数据类型",
            y_title="推理延迟 (ms)",
            color=chart.colors[2],
            height=350,
        )
        chart.display_chart(fig_speed)

    # ---- 交互式量化估算 ----
    st.markdown("#### 交互式量化估算")
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "选择模型",
            list(MODEL_QUANT_DATA.keys()),
            key="quant_model",
        )
    with col2:
        selected_bit = st.selectbox(
            "目标位宽",
            [32, 16, 8, 4],
            format_func=lambda x: f"FP{x}" if x >= 16 else f"INT{x}",
            key="quant_bit",
        )

    result = compute_quantization_impact(selected_model, selected_bit)
    info = MODEL_QUANT_DATA[selected_model]

    st.markdown(
        f"""
        **{selected_model}** 量化结果：

        | 指标 | FP32 | {('FP' if selected_bit >= 16 else 'INT') + str(selected_bit)} |
        |:-----|:----:|:----:|
        | 模型大小 | {info['fp32_size_mb']:.1f} MB | {result['size_mb']:.1f} MB |
        | 大小比例 | 1.00x | {result['size_ratio']:.3f}x |
        | 精度 | {info['fp32_acc']:.2f}% | {result['accuracy']:.2f}% |
        """
    )

    if info["fp32_acc"] > 0:
        acc_drop = info["fp32_acc"] - result["accuracy"]
        if acc_drop < 0.5:
            st.success(f"精度损失仅 {acc_drop:.2f}%，几乎无损量化")
        elif acc_drop < 2.0:
            st.warning(f"精度损失 {acc_drop:.2f}%，在可接受范围内")
        else:
            st.error(f"精度损失 {acc_drop:.2f}%，需要谨慎评估")


# ============================================================
# Section 3: 知识蒸馏
# ============================================================

def _section_distillation(chart: ChartBuilder):
    """知识蒸馏分析"""
    st.header("知识蒸馏 Knowledge Distillation")
    st.markdown(
        """
        知识蒸馏通过让一个小模型（Student）学习大模型（Teacher）的软标签输出，
        实现模型压缩。核心思想是温度缩放的 softmax 提供更丰富的类间关系信息。
        """
    )

    # ---- Teacher-Student 架构概念 ----
    st.markdown("#### Teacher-Student 架构")
    st.markdown(
        """
        ```
        Teacher Model (大模型)
           |
           | logits (z_t)
           |
           v
        Temperature Scaling: softmax(z_t / T)
           |
           | soft targets (软标签)
           |
           +---> Student Model (小模型)
                    |
                    | logits (z_s)
                    |
                    v
                 softmax(z_s / T)  -->  KL Divergence (L_soft)
                 softmax(z_s)      -->  Cross Entropy (L_hard)
        ```
        """
    )

    # ---- 温度缩放公式 ----
    st.markdown("#### 温度缩放公式")
    st.markdown(
        r"""
        标准 softmax：
        $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

        温度缩放 softmax：
        $$p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

        - $T = 1$：标准 softmax，输出接近 one-hot
        - $T > 1$：软化分布，暴露类间关系（"暗知识"）
        - $T \to \infty$：趋近均匀分布
        """
    )

    # ---- 交互式温度演示 ----
    st.markdown("#### 交互式温度缩放演示")

    # 示例 logits（模拟一个 5 类分类器的输出）
    example_logits = (3.0, 1.0, 0.5, -1.0, -2.0)
    class_names = ["猫", "狗", "兔", "鸟", "鱼"]

    temperature = st.slider(
        "温度 T", min_value=1, max_value=20, value=5, step=1,
        key="distill_temp",
    )

    orig_dist, soft_dist = compute_temperature_distribution(
        example_logits, temperature
    )

    # 显示分布对比
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**原始分布 (T=1)**")
        for name, prob in zip(class_names, orig_dist):
            st.markdown(f"- {name}: {prob:.4f} ({prob*100:.1f}%)")

    with col2:
        st.markdown(f"**软化分布 (T={temperature})**")
        for name, prob in zip(class_names, soft_dist):
            st.markdown(f"- {name}: {prob:.4f} ({prob*100:.1f}%)")

    # 温度 vs 分布变化折线图
    temps = list(range(1, 21))
    all_dists = []
    for t in temps:
        _, dist = compute_temperature_distribution(example_logits, t)
        all_dists.append(dist)

    # 转置：每个类一条线
    class_dists = [list(d) for d in zip(*all_dists)]

    fig_temp = chart.create_line_chart(
        x_data=temps,
        y_data=class_dists,
        title="温度 vs 概率分布变化",
        x_title="温度 T",
        y_title="概率",
        line_names=class_names,
        height=400,
    )
    chart.display_chart(fig_temp)

    # ---- 损失函数 ----
    st.markdown("#### 蒸馏损失函数")
    st.markdown(
        r"""
        总损失：
        $$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot \mathcal{L}_{\text{soft}} \cdot T^2$$

        - $\mathcal{L}_{\text{hard}}$：学生模型输出与真实标签的交叉熵
        - $\mathcal{L}_{\text{soft}}$：学生软标签与教师软标签的 KL 散度
        - $\alpha$：硬标签权重（通常 0.1~0.5）
        - $T^2$：温度平方补偿（因为梯度随 $T^2$ 缩小）
        """
    )

    # ---- 交互式损失计算 ----
    st.markdown("#### 交互式损失计算")
    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider(
            "alpha (硬标签权重)", min_value=0.0, max_value=1.0,
            value=0.1, step=0.05, key="distill_alpha",
        )
    with col2:
        temp = st.slider(
            "温度 T", min_value=1, max_value=20,
            value=4, step=1, key="distill_temp_loss",
        )
    with col3:
        st.markdown(
            """
            **固定值**：
            - L_hard = 2.30
            - L_soft = 0.85
            """
        )

    total, hard_c, soft_c = compute_distillation_loss(
        alpha, temp, 2.30, 0.85
    )

    st.markdown(
        f"""
        **损失分解**：
        - 硬标签贡献：$\alpha \\cdot L_{{hard}}$ = {alpha} x 2.30 = **{hard_c:.4f}**
        - 软标签贡献：$(1-\\alpha) \\cdot L_{{soft}} \\cdot T^2$ = {1-alpha:.2f} x 0.85 x {temp}^2 = **{soft_c:.4f}**
        - **总损失** = **{total:.4f}**
        """
    )

    # alpha 和 T 对损失的影响图
    alphas = np.arange(0.0, 1.05, 0.05).tolist()
    total_losses = []
    hard_losses = []
    soft_losses = []
    for a in alphas:
        t_l, h_l, s_l = compute_distillation_loss(a, temp, 2.30, 0.85)
        total_losses.append(t_l)
        hard_losses.append(h_l)
        soft_losses.append(s_l)

    fig_loss = chart.create_line_chart(
        x_data=[f"{a:.2f}" for a in alphas],
        y_data=[hard_losses, soft_losses, total_losses],
        title=f"alpha vs 损失分解 (T={temp})",
        x_title="alpha",
        y_title="损失值",
        line_names=["L_hard 贡献", "L_soft 贡献", "总损失"],
        height=400,
    )
    chart.display_chart(fig_loss)


# ============================================================
# Section 4: 结构化剪枝
# ============================================================

def _section_pruning(chart: ChartBuilder):
    """结构化剪枝分析"""
    st.header("结构化剪枝 Structured Pruning")
    st.markdown(
        """
        结构化剪枝移除整个神经元或卷积核，直接减小模型体积并获得实际推理加速。
        与非结构化剪枝（移除单个权重）不同，结构化剪枝不需要稀疏矩阵支持。
        """
    )

    # ---- 剪枝比例 vs 精度 ----
    ratios_pct = [f"{r*100:.0f}%" for r in PRUNING_DATA["ratios"]]

    fig_prune = chart.create_line_chart(
        x_data=ratios_pct,
        y_data=[PRUNING_DATA["magnitude_acc"], PRUNING_DATA["gradient_acc"]],
        title="剪枝比例 vs 精度 (ResNet-50, ImageNet)",
        x_title="剪枝比例",
        y_title="Top-1 Accuracy (%)",
        line_names=["Magnitude Pruning", "Gradient-based Pruning"],
        height=400,
    )
    chart.display_chart(fig_prune)

    # ---- 剪枝比例 vs FLOPs ----
    fig_flops = chart.create_line_chart(
        x_data=ratios_pct,
        y_data=[PRUNING_DATA["flops_reduction"]],
        title="剪枝比例 vs FLOPs 减少",
        x_title="剪枝比例",
        y_title="FLOPs 减少比例",
        line_names=["FLOPs 减少"],
        height=350,
    )
    chart.display_chart(fig_flops)

    # ---- Magnitude vs Gradient 对比 ----
    st.markdown("#### Magnitude Pruning vs Gradient-based Pruning")
    st.markdown(
        """
        | 方法 | 原理 | 优点 | 缺点 |
        |:-----|:-----|:-----|:-----|
        | Magnitude Pruning | 移除绝对值最小的权重 | 实现简单、无需训练数据 | 忽略权重对输出的实际影响 |
        | Gradient-based Pruning | 移除梯度最小的权重 | 更精确的冗余判断 | 需要前向-反向传播，计算开销大 |
        """
    )

    # ---- 交互式剪枝估算 ----
    st.markdown("#### 交互式剪枝估算")
    pruning_ratio = st.slider(
        "剪枝比例", min_value=0.0, max_value=0.9,
        value=0.5, step=0.05, key="prune_ratio",
    )
    prune_method = st.selectbox(
        "剪枝方法",
        ["magnitude", "gradient"],
        format_func=lambda x: "Magnitude Pruning" if x == "magnitude" else "Gradient-based Pruning",
        key="prune_method",
    )

    est = compute_pruning_estimate(pruning_ratio, prune_method)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预估精度", f"{est['accuracy']:.2f}%")
    with col2:
        st.metric("FLOPs 减少", f"{est['flops_reduction']*100:.1f}%")
    with col3:
        st.metric("模型大小比例", f"{est['size_ratio']*100:.1f}%")

    acc_drop = 76.15 - est["accuracy"]
    if acc_drop < 1.0:
        st.success(f"精度损失仅 {acc_drop:.2f}%，剪枝效果良好")
    elif acc_drop < 5.0:
        st.warning(f"精度损失 {acc_drop:.2f}%，建议微调恢复")
    else:
        st.error(f"精度损失 {acc_drop:.2f}%，剪枝过于激进")


# ============================================================
# Section 5: 压缩流水线
# ============================================================

def _section_pipeline(chart: ChartBuilder):
    """压缩流水线"""
    st.header("压缩流水线 Compression Pipeline")
    st.markdown(
        """
        在实际部署中，多种压缩技术可以组合使用，形成流水线：
        **Pruning -> Quantization -> Distillation**

        每一步在前一步的基础上进一步压缩，同时蒸馏帮助恢复精度损失。
        """
    )

    # ---- 参数控制 ----
    col1, col2, col3 = st.columns(3)
    with col1:
        pipe_prune_ratio = st.slider(
            "剪枝比例", min_value=0.0, max_value=0.7,
            value=0.3, step=0.05, key="pipe_prune",
        )
    with col2:
        pipe_quant_bit = st.selectbox(
            "量化位宽",
            [32, 16, 8, 4],
            format_func=lambda x: f"FP{x}" if x >= 16 else f"INT{x}",
            key="pipe_quant",
        )
    with col3:
        pipe_alpha = st.slider(
            "蒸馏 alpha", min_value=0.0, max_value=1.0,
            value=0.3, step=0.05, key="pipe_alpha",
        )

    pipe_result = compute_pipeline_compression(
        pipe_prune_ratio, pipe_quant_bit, pipe_alpha,
    )

    # ---- 步骤可视化 ----
    st.markdown("#### 流水线步骤详情")

    steps_data = [
        {
            "步骤": "1. Pruning",
            "操作": f"剪枝 {pipe_prune_ratio*100:.0f}%",
            "累积压缩比": f"{pipe_result['pruning_compression']:.2f}x",
            "精度": f"{pipe_result['pruning_acc']:.2f}%",
        },
        {
            "步骤": "2. Quantization",
            "操作": f"量化到 {('FP' if pipe_quant_bit >= 16 else 'INT') + str(pipe_quant_bit)}",
            "累积压缩比": f"{pipe_result['quant_compression']:.2f}x",
            "精度": f"{pipe_result['post_quant_acc']:.2f}%",
        },
        {
            "步骤": "3. Distillation",
            "操作": f"蒸馏恢复 (alpha={pipe_alpha:.2f})",
            "累积压缩比": f"{pipe_result['cumulative_compression']:.2f}x",
            "精度": f"{pipe_result['final_acc']:.2f}%",
        },
    ]

    df = pd.DataFrame(steps_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- 累积效果图表 ----
    st.markdown("#### 累积压缩比与精度变化")

    fig_pipe = chart.create_subplots(
        rows=1, cols=2,
        subplot_titles=["累积压缩比", "精度变化"],
        height=400,
    )

    fig_pipe.add_trace(
        go.Bar(
            x=["原始", "Pruning", "Quantization", "蒸馏后"],
            y=[1.0, pipe_result["pruning_compression"],
               pipe_result["quant_compression"],
               pipe_result["cumulative_compression"]],
            marker_color=[chart.colors[0], chart.colors[1],
                          chart.colors[2], chart.colors[3]],
            text=[f"{v:.1f}x" for v in [1.0, pipe_result["pruning_compression"],
                                         pipe_result["quant_compression"],
                                         pipe_result["cumulative_compression"]]],
            textposition="auto",
        ),
        row=1, col=1,
    )

    fig_pipe.add_trace(
        go.Bar(
            x=["原始", "Pruning后", "Quantization后", "蒸馏恢复后"],
            y=[76.15, pipe_result["pruning_acc"],
               pipe_result["post_quant_acc"],
               pipe_result["final_acc"]],
            marker_color=[chart.colors[0], chart.colors[1],
                          chart.colors[2], chart.colors[3]],
            text=[f"{v:.1f}%" for v in [76.15, pipe_result["pruning_acc"],
                                        pipe_result["post_quant_acc"],
                                        pipe_result["final_acc"]]],
            textposition="auto",
        ),
        row=1, col=2,
    )

    fig_pipe.update_layout(height=420)
    chart.display_chart(fig_pipe)

    # ---- 总结 ----
    total_acc_drop = 76.15 - pipe_result["final_acc"]
    st.markdown(
        f"""
        #### 流水线总结

        | 指标 | 值 |
        |:-----|:---|
        | 最终压缩比 | **{pipe_result['cumulative_compression']:.2f}x** |
        | 最终精度 | **{pipe_result['final_acc']:.2f}%** |
        | 精度损失 | **{total_acc_drop:.2f}%** |
        | 蒸馏恢复 | **+{pipe_result['distillation_recovery']:.2f}%** |
        """
    )

    if total_acc_drop < 2.0:
        st.success("压缩效果优秀，精度损失在可接受范围内")
    elif total_acc_drop < 5.0:
        st.warning("压缩效果良好，建议适当降低剪枝比例或提高量化位宽")
    else:
        st.error("精度损失较大，建议减少压缩强度或增加蒸馏训练轮次")


# ============================================================
# 主函数
# ============================================================


def model_compression_tab(chinese_supported=True):
    """模型压缩分析主函数

    Args:
        chinese_supported: 是否支持中文
    """
    chart = ChartBuilder()

    # 使用 tabs 组织各节
    tab_names = [
        "概览 Overview",
        "量化 Quantization",
        "蒸馏 Distillation",
        "剪枝 Pruning",
        "流水线 Pipeline",
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

    with tab1:
        _section_overview(chart)

    with tab2:
        _section_quantization(chart)

    with tab3:
        _section_distillation(chart)

    with tab4:
        _section_pruning(chart)

    with tab5:
        _section_pipeline(chart)


if __name__ == "__main__":
    model_compression_tab()
