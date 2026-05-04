"""
注意力机制分析 (Attention Mechanism Analysis)

深入分析 Attention 机制的核心原理与变体:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention vs Cross-Attention
- Attention Pattern Analysis
- Positional Encoding
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.visualization.chart_utils import ChartBuilder


# ============================================================
# 缓存计算函数
# ============================================================


@st.cache_data
def compute_scaled_dot_product_attention(seq_len, d_k, seed=42):
    """
    计算Scaled Dot-Product Attention的每一步

    Args:
        seq_len: 序列长度
        d_k: Key的维度
        seed: 随机种子

    Returns:
        dict: 包含Q, K, V, scores, scaled_scores, attn_weights, output
    """
    rng = np.random.RandomState(seed)

    # 生成随机 Q, K, V 矩阵
    Q = rng.randn(seq_len, d_k)
    K = rng.randn(seq_len, d_k)
    V = rng.randn(seq_len, d_k)

    # Step 1: QK^T
    scores = Q @ K.T  # [seq_len, seq_len]

    # Step 2: 缩放
    scale_factor = np.sqrt(d_k)
    scaled_scores = scores / scale_factor

    # Step 3: Softmax
    # 数值稳定: 减去最大值
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: 乘以 V
    output = attn_weights @ V  # [seq_len, d_k]

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "scaled_scores": scaled_scores,
        "attn_weights": attn_weights,
        "output": output,
        "scale_factor": scale_factor,
    }


@st.cache_data
def compute_multi_head_attention(seq_len, d_model, num_heads, seed=42):
    """
    计算Multi-Head Attention

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        num_heads: 注意力头数
        seed: 随机种子

    Returns:
        dict: 包含各头的注意力权重和最终输出
    """
    rng = np.random.RandomState(seed)
    d_k = d_model // num_heads

    # 生成随机输入
    X = rng.randn(seq_len, d_model)

    # 生成投影矩阵 W_q, W_k, W_v
    W_q = rng.randn(d_model, d_model) * 0.02
    W_k = rng.randn(d_model, d_model) * 0.02
    W_v = rng.randn(d_model, d_model) * 0.02
    W_o = rng.randn(d_model, d_model) * 0.02

    # 投影
    Q = X @ W_q  # [seq_len, d_model]
    K = X @ W_k
    V = X @ W_v

    # 按头拆分
    Q_heads = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # [num_heads, seq_len, d_k]
    K_heads = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    # 各头计算注意力
    head_weights = []
    head_outputs = []
    for h in range(num_heads):
        scores = Q_heads[h] @ K_heads[h].T / np.sqrt(d_k)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_w = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        head_out = attn_w @ V_heads[h]
        head_weights.append(attn_w)
        head_outputs.append(head_out)

    # 拼接各头输出
    concat = np.concatenate(head_outputs, axis=-1)  # [seq_len, d_model]

    # 线性投影
    final_output = concat @ W_o

    return {
        "head_weights": head_weights,
        "head_outputs": head_outputs,
        "concat": concat,
        "final_output": final_output,
        "d_k": d_k,
    }


@st.cache_data
def compute_self_cross_attention(seq_len, d_k, seed=42):
    """
    计算 Self-Attention 和 Cross-Attention 并对比

    Args:
        seq_len: 序列长度
        d_k: Key维度
        seed: 随机种子

    Returns:
        dict: self_attn 和 cross_attn 的权重
    """
    rng = np.random.RandomState(seed)

    # Self-Attention: Q=K=V 来自同一序列
    X = rng.randn(seq_len, d_k)
    Q_self = K_self = V_self = X
    scores_self = Q_self @ K_self.T / np.sqrt(d_k)
    exp_s = np.exp(scores_self - np.max(scores_self, axis=-1, keepdims=True))
    self_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    # Cross-Attention: Q 来自 decoder, K=V 来自 encoder
    encoder_out = rng.randn(seq_len, d_k)
    decoder_out = rng.randn(seq_len, d_k)
    Q_cross = decoder_out
    K_cross = V_cross = encoder_out
    scores_cross = Q_cross @ K_cross.T / np.sqrt(d_k)
    exp_c = np.exp(scores_cross - np.max(scores_cross, axis=-1, keepdims=True))
    cross_weights = exp_c / np.sum(exp_c, axis=-1, keepdims=True)

    return {
        "self_weights": self_weights,
        "cross_weights": cross_weights,
    }


@st.cache_data
def compute_attention_masks(seq_len, mask_type, window_size=3, block_size=4, sparsity=0.5, seed=42):
    """
    生成不同类型的注意力掩码 (Attention Mask)

    Args:
        seq_len: 序列长度
        mask_type: 掩码类型 ('diagonal', 'block', 'random_sparse', 'causal', 'full')
        window_size: 局部窗口大小 (用于 diagonal)
        block_size: 块大小 (用于 block)
        sparsity: 稀疏度 (用于 random_sparse)
        seed: 随机种子

    Returns:
        np.ndarray: 注意力掩码矩阵 (1=允许注意, 0=屏蔽)
    """
    rng = np.random.RandomState(seed)

    if mask_type == "full":
        return np.ones((seq_len, seq_len))

    elif mask_type == "causal":
        # 因果掩码 (Causal Mask): 只能看到当前位置及之前
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask

    elif mask_type == "diagonal":
        # 局部窗口掩码 (Local/Sliding Window)
        mask = np.zeros((seq_len, seq_len))
        half_w = window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = 1.0
        return mask

    elif mask_type == "block":
        # 块状掩码 (Block Sparse)
        mask = np.zeros((seq_len, seq_len))
        num_blocks = seq_len // block_size
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            mask[start:end, start:end] = 1.0
        # 处理余数
        remainder = seq_len % block_size
        if remainder > 0:
            start = num_blocks * block_size
            mask[start:, start:] = 1.0
        return mask

    elif mask_type == "random_sparse":
        # 随机稀疏掩码
        mask = np.zeros((seq_len, seq_len))
        num_nonzero = int(seq_len * seq_len * (1 - sparsity))
        indices = rng.choice(seq_len * seq_len, num_nonzero, replace=False)
        mask.flat[indices] = 1.0
        return mask

    return np.ones((seq_len, seq_len))


@st.cache_data
def compute_positional_encoding(seq_len, d_model, seed=42):
    """
    计算正弦位置编码 (Sinusoidal Positional Encoding)

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        seed: 随机种子 (用于生成学习式位置编码)

    Returns:
        dict: sinusoidal 编码矩阵和 learned 编码矩阵
    """
    # 正弦位置编码
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # [seq_len, 1]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度

    # 学习式位置编码 (随机初始化模拟)
    rng = np.random.RandomState(seed)
    learned_pe = rng.randn(seq_len, d_model) * 0.02

    return {
        "sinusoidal": pe,
        "learned": learned_pe,
    }


# ============================================================
# Section 1: Scaled Dot-Product Attention
# ============================================================


def section_scaled_dot_product_attention(chart: ChartBuilder):
    """Section 1: Scaled Dot-Product Attention 核心公式"""

    st.subheader("1. Scaled Dot-Product Attention 缩放点积注意力")

    st.markdown(
        r"""
        ### 核心公式

        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

        **公式拆解**:
        - $Q$ (Query): 查询矩阵，形状 $[n, d_k]$
        - $K$ (Key): 键矩阵，形状 $[n, d_k]$
        - $V$ (Value): 值矩阵，形状 $[n, d_k]$
        - $d_k$: Key 的维度，用于缩放以防止梯度消失
        - $\sqrt{d_k}$: 缩放因子，确保点积的方差为 1
        """
    )

    st.markdown(
        """
        **为什么需要缩放?**

        当 $d_k$ 较大时，$QK^T$ 的点积结果会很大，导致 softmax 进入饱和区域，
        梯度趋近于零。除以 $\sqrt{d_k}$ 可以将方差归一化到 1。
        """
    )

    st.markdown("---")

    # 交互式参数设置
    col1, col2, col3 = st.columns(3)

    with col1:
        seq_len = st.slider(
            "序列长度 (seq_len)",
            min_value=4,
            max_value=16,
            value=6,
            step=1,
            key="attn_seq_len",
        )

    with col2:
        d_k = st.slider(
            "Key维度 (d_k)",
            min_value=4,
            max_value=16,
            value=8,
            step=2,
            key="attn_d_k",
        )

    with col3:
        seed = st.slider(
            "随机种子",
            min_value=0,
            max_value=100,
            value=42,
            step=1,
            key="attn_seed",
        )

    # 计算注意力
    result = compute_scaled_dot_product_attention(seq_len, d_k, seed)

    st.info(
        f"""
        **矩阵形状**:
        - Q: [{seq_len}, {d_k}], K: [{seq_len}, {d_k}], V: [{seq_len}, {d_k}]
        - QK^T: [{seq_len}, {seq_len}]
        - 缩放因子: $\sqrt{{{d_k}}}$ = {result['scale_factor']:.4f}
        - 输出: [{seq_len}, {d_k}]
        """
    )

    # Step 1: QK^T
    st.markdown("#### Step 1: 计算点积 $QK^T$")
    fig_scores = chart.create_heatmap(
        result["scores"],
        title="QK^T 点积矩阵 (未缩放)",
        x_title="Key 位置",
        y_title="Query 位置",
        colorscale="RdBu",
        height=350,
    )
    chart.display_chart(fig_scores)

    # Step 2: 缩放
    st.markdown(f"#### Step 2: 缩放 $QK^T / \\sqrt{{d_k}}$ = $QK^T / {result['scale_factor']:.4f}$")
    fig_scaled = chart.create_heatmap(
        result["scaled_scores"],
        title="缩放后的得分矩阵",
        x_title="Key 位置",
        y_title="Query 位置",
        colorscale="RdBu",
        height=350,
    )
    chart.display_chart(fig_scaled)

    # Step 3: Softmax
    st.markdown("#### Step 3: Softmax 归一化")
    fig_attn = chart.create_heatmap(
        result["attn_weights"],
        title="Softmax 注意力权重 (每行和为1)",
        x_title="Key 位置",
        y_title="Query 位置",
        colorscale="YlOrRd",
        height=350,
    )
    chart.display_chart(fig_attn)

    # 验证每行和为1
    row_sums = result["attn_weights"].sum(axis=1)
    st.markdown(
        f"""
        **验证**: 每行 softmax 输出之和应为 1:
        行和 = [{', '.join(f'{s:.6f}' for s in row_sums)}]
        """
    )

    # Step 4: 最终输出
    st.markdown("#### Step 4: 加权求和 $\text{Attention Weights} \times V$")
    fig_output = chart.create_heatmap(
        result["output"],
        title="最终输出矩阵 (Attention Output)",
        x_title="Value 维度",
        y_title="序列位置",
        colorscale="Viridis",
        height=350,
    )
    chart.display_chart(fig_output)

    st.success(
        """
        **要点总结**:
        - 注意力权重的每一行代表一个 Query 对所有 Key 的关注程度
        - 权重通过 softmax 归一化，每行和为 1
        - 最终输出是 Value 矩阵的加权和，权重由注意力分布决定
        - 缩放因子 $\sqrt{d_k}$ 是防止 softmax 饱和的关键
        """
    )


# ============================================================
# Section 2: Multi-Head Attention
# ============================================================


def section_multi_head_attention(chart: ChartBuilder):
    """Section 2: Multi-Head Attention 多头注意力"""

    st.subheader("2. Multi-Head Attention 多头注意力")

    st.markdown(
        r"""
        ### 核心思想

        $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

        其中每个头:
        $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

        **多头注意力的优势**:
        - 不同的头可以关注不同的位置和语义信息
        - 类似于 CNN 中多个卷积核提取不同特征
        - 每个头的维度 $d_k = d_{model} / h$，总计算量与单头相近
        """
    )

    st.markdown("---")

    # 交互式参数
    col1, col2, col3 = st.columns(3)

    with col1:
        mha_seq_len = st.slider(
            "序列长度",
            min_value=4,
            max_value=12,
            value=6,
            step=1,
            key="mha_seq_len",
        )

    with col2:
        d_model = st.slider(
            "模型维度 (d_model)",
            min_value=8,
            max_value=32,
            value=16,
            step=4,
            key="mha_d_model",
        )

    with col3:
        num_heads = st.slider(
            "注意力头数 (num_heads)",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            key="mha_num_heads",
        )

    # 确保 d_model 能被 num_heads 整除
    if d_model % num_heads != 0:
        st.warning(
            f"⚠️ d_model ({d_model}) 不能被 num_heads ({num_heads}) 整除! "
            f"已自动调整 d_model 为 {num_heads * (d_model // num_heads)}"
        )
        d_model = num_heads * (d_model // num_heads)

    d_k = d_model // num_heads

    st.info(
        f"""
        **配置**:
        - 序列长度: {mha_seq_len}
        - 模型维度: {d_model}
        - 注意力头数: {num_heads}
        - 每头维度: $d_k$ = {d_model} / {num_heads} = {d_k}
        """
    )

    # 计算多头注意力
    result = compute_multi_head_attention(mha_seq_len, d_model, num_heads)

    # 可视化各头的注意力模式
    st.markdown("#### 各头的注意力模式 (Attention Patterns)")

    heads_to_show = min(num_heads, 4)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Head {i + 1}" for i in range(heads_to_show)],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    for idx in range(heads_to_show):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.add_trace(
            go.Heatmap(
                z=result["head_weights"][idx],
                colorscale="YlOrRd",
                showscale=(idx == 0),
                colorbar=dict(x=1.02 if idx == 0 else None, len=0.8),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Multi-Head Attention: 各头注意力权重对比",
        height=500,
        margin=dict(t=80, b=40),
    )
    chart.display_chart(fig)

    st.markdown(
        """
        **观察**:
        - 不同的头学习到了不同的注意力模式
        - 某些头可能关注局部 (对角线亮)，某些关注全局
        - 这就是多头注意力的核心优势: 多角度建模序列关系
        """
    )

    # 拼接和投影可视化
    st.markdown("#### 拼接 (Concat) 与线性投影 (Linear Projection)")

    col1, col2 = st.columns(2)

    with col1:
        fig_concat = chart.create_heatmap(
            result["concat"],
            title=f"拼接后矩阵 Concat [{mha_seq_len}, {d_model}]",
            x_title="维度",
            y_title="序列位置",
            colorscale="Viridis",
            height=350,
        )
        chart.display_chart(fig_concat)

    with col2:
        fig_output = chart.create_heatmap(
            result["final_output"],
            title=f"线性投影后输出 [{mha_seq_len}, {d_model}]",
            x_title="维度",
            y_title="序列位置",
            colorscale="Viridis",
            height=350,
        )
        chart.display_chart(fig_output)

    st.success(
        f"""
        **Multi-Head Attention 计算流程**:
        1. 输入 X [{mha_seq_len}, {d_model}] 通过 W_q, W_k, W_v 投影
        2. 拆分为 {num_heads} 个头，每头维度 {d_k}
        3. 各头独立计算 Scaled Dot-Product Attention
        4. 拼接 {num_heads} 个头的输出 → [{mha_seq_len}, {d_model}]
        5. 通过 W_o 线性投影得到最终输出
        """
    )


# ============================================================
# Section 3: Self-Attention vs Cross-Attention
# ============================================================


def section_self_vs_cross_attention(chart: ChartBuilder):
    """Section 3: Self-Attention vs Cross-Attention 对比"""

    st.subheader("3. Self-Attention vs Cross-Attention 对比")

    st.markdown(
        r"""
        ### Self-Attention (自注意力)

        $$\text{Self-Attention}: Q = K = V = X$$

        序列中的每个位置同时关注自身序列的所有位置。用于 Encoder 和 Decoder 的自注意力层。

        ### Cross-Attention (交叉注意力)

        $$\text{Cross-Attention}: Q = X_{decoder}, \quad K = V = X_{encoder}$$

        Query 来自一个序列 (Decoder)，Key 和 Value 来自另一个序列 (Encoder)。
        用于 Decoder 中关注 Encoder 的输出。
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        sc_seq_len = st.slider(
            "序列长度",
            min_value=4,
            max_value=12,
            value=6,
            step=1,
            key="sc_seq_len",
        )

    with col2:
        sc_d_k = st.slider(
            "维度 (d_k)",
            min_value=4,
            max_value=16,
            value=8,
            step=2,
            key="sc_d_k",
        )

    result = compute_self_cross_attention(sc_seq_len, sc_d_k)

    # 并排对比
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Self-Attention 注意力权重")
        st.markdown("$Q = K = V$ (来自同一序列)")
        fig_self = chart.create_heatmap(
            result["self_weights"],
            title="Self-Attention 权重",
            x_title="Key 位置",
            y_title="Query 位置",
            colorscale="YlOrRd",
            height=400,
        )
        chart.display_chart(fig_self)

    with col2:
        st.markdown("#### Cross-Attention 注意力权重")
        st.markdown("$Q$ 来自 Decoder, $K=V$ 来自 Encoder")
        fig_cross = chart.create_heatmap(
            result["cross_weights"],
            title="Cross-Attention 权重",
            x_title="Encoder 位置",
            y_title="Decoder 位置",
            colorscale="YlOrRd",
            height=400,
        )
        chart.display_chart(fig_cross)

    st.markdown("---")

    # 对比表格
    st.markdown("#### Self-Attention vs Cross-Attention 详细对比")

    comparison_table = """
    | 特性 | Self-Attention | Cross-Attention |
    |:-----|:---------------|:-----------------|
    | **Q 来源** | 输入序列 X | Decoder 输出 |
    | **K 来源** | 输入序列 X | Encoder 输出 |
    | **V 来源** | 输入序列 X | Encoder 输出 |
    | **注意力矩阵形状** | $[n, n]$ | $[m, n]$ (m=decoder长度, n=encoder长度) |
    | **典型应用** | BERT, GPT Encoder层 | Transformer Decoder层, 图文匹配 |
    | **作用** | 捕获序列内部依赖 | 对齐两个不同序列 |
    | **对称性** | 对称矩阵 (Q=K) | 非对称矩阵 (Q≠K) |
    """
    st.markdown(comparison_table)

    st.success(
        """
        **关键理解**:
        - Self-Attention: "我需要关注自己序列中的哪些部分?" → 序列内部关系建模
        - Cross-Attention: "我需要关注源序列中的哪些部分?" → 跨序列对齐
        - Transformer Encoder 只用 Self-Attention
        - Transformer Decoder 同时使用 Self-Attention 和 Cross-Attention
        """
    )


# ============================================================
# Section 4: Attention Pattern Analysis
# ============================================================


def section_attention_patterns(chart: ChartBuilder):
    """Section 4: Attention Pattern Analysis 注意力模式分析"""

    st.subheader("4. Attention Pattern Analysis 注意力模式分析")

    st.markdown(
        """
        标准的 Self-Attention 计算复杂度为 $O(n^2)$，当序列长度 $n$ 很大时计算开销巨大。
        各种稀疏注意力模式 (Sparse Attention) 通过限制每个位置只能关注部分位置来降低复杂度。
        """
    )

    st.markdown("---")

    seq_len_pattern = st.slider(
        "序列长度",
        min_value=8,
        max_value=32,
        value=16,
        step=1,
        key="pattern_seq_len",
    )

    mask_types = {
        "full": "全连接 (Full Attention) - $O(n^2)$",
        "causal": "因果掩码 (Causal Mask) - 下三角",
        "diagonal": "局部窗口 (Sliding Window) - 带宽限制",
        "block": "块状稀疏 (Block Sparse) - 分块注意力",
        "random_sparse": "随机稀疏 (Random Sparse) - 随机采样",
    }

    selected_mask = st.selectbox(
        "选择注意力模式",
        options=list(mask_types.keys()),
        format_func=lambda x: mask_types[x],
        key="pattern_mask_type",
    )

    # 额外参数
    extra_params = {}
    if selected_mask == "diagonal":
        extra_params["window_size"] = st.slider(
            "窗口大小", min_value=1, max_value=seq_len_pattern, value=5, step=2,
            key="pattern_window_size",
        )
    elif selected_mask == "block":
        extra_params["block_size"] = st.slider(
            "块大小", min_value=2, max_value=seq_len_pattern // 2, value=4, step=2,
            key="pattern_block_size",
        )
    elif selected_mask == "random_sparse":
        extra_params["sparsity"] = st.slider(
            "稀疏度 (0=全连接, 1=全屏蔽)", min_value=0.1, max_value=0.9, value=0.7, step=0.05,
            key="pattern_sparsity",
        )

    # 计算掩码
    mask = compute_attention_masks(seq_len_pattern, selected_mask, **extra_params)

    # 可视化掩码
    fig_mask = chart.create_heatmap(
        mask,
        title=f"注意力掩码: {mask_types[selected_mask].split(' - ')[0]}",
        x_title="Key 位置",
        y_title="Query 位置",
        colorscale="Greys",
        height=400,
    )
    chart.display_chart(fig_mask)

    # 计算复杂度
    total_elements = seq_len_pattern * seq_len_pattern
    active_elements = int(np.sum(mask))
    flops_ratio = active_elements / total_elements

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总元素数", f"{total_elements}")
    with col2:
        st.metric("活跃元素数", f"{active_elements}")
    with col3:
        st.metric("FLOPs 比例", f"{flops_ratio:.1%}")

    st.markdown("---")

    # 所有模式对比
    st.markdown("#### 所有注意力模式对比")

    fig_all = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=["Full", "Causal", "Sliding Window", "Block Sparse", "Random Sparse", "FLOPs 对比"],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    all_masks = {
        "full": {},
        "causal": {},
        "diagonal": {"window_size": 5},
        "block": {"block_size": 4},
        "random_sparse": {"sparsity": 0.7},
    }

    flops_data = []
    names = []

    for idx, (mtype, params) in enumerate(all_masks.items()):
        row = idx // 3 + 1
        col = idx % 3 + 1
        m = compute_attention_masks(seq_len_pattern, mtype, **params)
        fig_all.add_trace(
            go.Heatmap(
                z=m,
                colorscale="Greys",
                showscale=False,
                zmin=0,
                zmax=1,
            ),
            row=row,
            col=col,
        )
        ratio = np.sum(m) / (seq_len_pattern * seq_len_pattern)
        flops_data.append(ratio)
        names.append(mtype.replace("_", "\n"))

    # FLOPs 对比柱状图
    fig_all.add_trace(
        go.Bar(
            x=names,
            y=flops_data,
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            text=[f"{r:.0%}" for r in flops_data],
            textposition="auto",
        ),
        row=2,
        col=3,
    )

    fig_all.update_layout(
        title="注意力模式全景对比",
        height=600,
        margin=dict(t=80, b=40),
    )
    chart.display_chart(fig_all)

    st.success(
        """
        **注意力模式总结**:
        - **Full Attention**: 所有位置互相注意，$O(n^2)$ 复杂度，最精确但最慢
        - **Causal Mask**: 只能看到之前的位置，用于自回归生成 (GPT)
        - **Sliding Window**: 只关注局部窗口，$O(n \cdot w)$，适合长序列
        - **Block Sparse**: 分块注意力，块内全连接，块间稀疏
        - **Random Sparse**: 随机选择注意位置，兼顾全局和局部信息
        """
    )


# ============================================================
# Section 5: Positional Encoding
# ============================================================


def section_positional_encoding(chart: ChartBuilder):
    """Section 5: Positional Encoding 位置编码"""

    st.subheader("5. Positional Encoding 位置编码")

    st.markdown(
        r"""
        ### 为什么需要位置编码?

        Transformer 的 Self-Attention 是排列不变的 (Permutation Invariant)，
        即打乱输入顺序不影响输出。为了让模型理解序列中 token 的位置信息，
        需要显式地注入位置编码。

        ### 正弦位置编码 (Sinusoidal Positional Encoding)

        $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

        $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

        其中 $pos$ 是位置索引，$i$ 是维度索引，$d_{model}$ 是模型维度。
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        pe_seq_len = st.slider(
            "序列长度",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            key="pe_seq_len",
        )

    with col2:
        pe_d_model = st.slider(
            "模型维度 (d_model)",
            min_value=16,
            max_value=128,
            value=64,
            step=8,
            key="pe_d_model",
        )

    result = compute_positional_encoding(pe_seq_len, pe_d_model)

    # 正弦位置编码热力图
    st.markdown("#### 正弦位置编码矩阵 (Sinusoidal Positional Encoding)")
    fig_pe = chart.create_heatmap(
        result["sinusoidal"],
        title=f"位置编码矩阵 [{pe_seq_len}, {pe_d_model}]",
        x_title="维度",
        y_title="位置",
        colorscale="RdBu",
        height=450,
    )
    chart.display_chart(fig_pe)

    st.markdown(
        """
        **观察**:
        - 低维度 (左侧): 高频振荡，变化快 → 捕获相邻位置的差异
        - 高维度 (右侧): 低频振荡，变化慢 → 捕获远距离位置的关系
        - 这种多尺度设计让模型能区分不同距离的位置关系
        """
    )

    # 选取几个维度展示波形
    st.markdown("#### 不同维度的编码波形")

    dims_to_show = [0, 1, pe_d_model // 4, pe_d_model // 4 + 1, pe_d_model // 2, pe_d_model // 2 + 1]
    dims_to_show = [d for d in dims_to_show if d < pe_d_model]

    fig_wave = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, dim in enumerate(dims_to_show):
        func_type = "sin" if dim % 2 == 0 else "cos"
        freq = 1.0 / (10000 ** (dim / pe_d_model))
        fig_wave.add_trace(
            go.Scatter(
                x=list(range(pe_seq_len)),
                y=result["sinusoidal"][:, dim],
                mode="lines",
                name=f"dim={dim} ({func_type}, freq={freq:.4f})",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig_wave.update_layout(
        title="不同维度的位置编码波形",
        xaxis_title="位置 (pos)",
        yaxis_title="编码值",
        height=400,
        **ChartBuilder.DEFAULT_LAYOUT,
    )
    chart.display_chart(fig_wave)

    st.markdown("---")

    # 正弦 vs 学习式位置编码对比
    st.markdown("#### 正弦编码 vs 学习式编码 (Sinusoidal vs Learned)")

    col1, col2 = st.columns(2)

    with col1:
        fig_sin = chart.create_heatmap(
            result["sinusoidal"],
            title="正弦位置编码 (Sinusoidal)",
            x_title="维度",
            y_title="位置",
            colorscale="RdBu",
            height=350,
        )
        chart.display_chart(fig_sin)

    with col2:
        fig_learned = chart.create_heatmap(
            result["learned"],
            title="学习式位置编码 (Learned, 随机初始化)",
            x_title="维度",
            y_title="位置",
            colorscale="RdBu",
            height=350,
        )
        chart.display_chart(fig_learned)

    # 对比表格
    st.markdown("#### 两种位置编码对比")

    pe_comparison = """
    | 特性 | 正弦位置编码 | 学习式位置编码 |
    |:-----|:------------|:---------------|
    | **参数** | 无额外参数 | 需要 $n \times d_{model}$ 个可学习参数 |
    | **外推性** | 好 (可推广到更长序列) | 差 (超出训练长度则失效) |
    | **灵活性** | 固定模式 | 可学习最优编码 |
    | **代表模型** | 原始 Transformer | BERT, GPT, ViT |
    | **相对位置** | 天然支持 (通过三角恒等式) | 需要额外设计 (如 RoPE) |
    | **训练数据需求** | 少 | 多 |
    """
    st.markdown(pe_comparison)

    st.success(
        """
        **位置编码要点**:
        - 正弦编码利用不同频率的正弦/余弦函数编码位置，无需学习参数
        - 学习式编码将位置作为可学习参数，通常效果更好但不具备外推性
        - 现代模型 (如 LLaMA) 常用旋转位置编码 (RoPE, Rotary Position Embedding)
        - 位置编码是 Transformer 理解序列顺序的关键组件
        """
    )


# ============================================================
# Main Tab Function
# ============================================================


def attention_analysis_tab(chinese_supported=True):
    """注意力机制分析主函数

    Args:
        chinese_supported: 是否支持中文显示
    """

    st.header("注意力机制分析 (Attention Mechanism Analysis)")
    st.markdown(
        """
        > **核心问题**: Attention 机制是如何工作的? 为什么它是 Transformer 的核心?
        > 它有哪些变体和优化方法?

        **探索内容**:
        - Scaled Dot-Product Attention 的逐步计算过程
        - Multi-Head Attention 的多头并行机制
        - Self-Attention 与 Cross-Attention 的区别
        - 不同注意力模式及其计算效率
        - 位置编码的作用与实现方式
        """
    )

    st.markdown("---")

    # 初始化 ChartBuilder
    chart = ChartBuilder()

    # Section 1: Scaled Dot-Product Attention
    with st.expander("1. Scaled Dot-Product Attention 缩放点积注意力", expanded=True):
        section_scaled_dot_product_attention(chart)

    st.markdown("---")

    # Section 2: Multi-Head Attention
    with st.expander("2. Multi-Head Attention 多头注意力", expanded=False):
        section_multi_head_attention(chart)

    st.markdown("---")

    # Section 3: Self-Attention vs Cross-Attention
    with st.expander("3. Self-Attention vs Cross-Attention 对比", expanded=False):
        section_self_vs_cross_attention(chart)

    st.markdown("---")

    # Section 4: Attention Pattern Analysis
    with st.expander("4. Attention Pattern Analysis 注意力模式分析", expanded=False):
        section_attention_patterns(chart)

    st.markdown("---")

    # Section 5: Positional Encoding
    with st.expander("5. Positional Encoding 位置编码", expanded=False):
        section_positional_encoding(chart)

    st.markdown("---")

    # 总结
    st.subheader("核心要点总结")

    st.markdown(
        """
        ### Attention 机制的五大核心概念

        1. **Scaled Dot-Product Attention**: 注意力的基本计算单元
           - $QK^T$ 计算相似度 → 缩放 → softmax → 加权求和
           - 缩放因子 $\sqrt{d_k}$ 防止梯度消失

        2. **Multi-Head Attention**: 多角度建模序列关系
           - 将 $d_{model}$ 拆分为多个头，并行计算
           - 不同头学习不同的注意力模式
           - 拼接后线性投影得到最终输出

        3. **Self vs Cross Attention**: 两种注意力范式的区别
           - Self: 同一序列内部的关系建模
           - Cross: 两个序列之间的对齐

        4. **Sparse Attention**: 降低 $O(n^2)$ 复杂度的策略
           - 局部窗口、块状稀疏、随机稀疏等
           - 在精度和效率之间取得平衡

        5. **Positional Encoding**: 为无序的 Attention 注入位置信息
           - 正弦编码: 无参数，可外推
           - 学习式编码: 更灵活，效果通常更好
        """
    )

    st.info(
        """
        **延伸阅读**:
        - 原始论文: "Attention Is All You Need" (Vaswani et al., 2017)
        - Longformer: 局部窗口 + 全局 token 的稀疏注意力
        - Linformer: 将注意力矩阵从 $O(n^2)$ 降为 $O(n)$
        - RoPE: 旋转位置编码，被 LLaMA 等现代模型广泛采用
        - Flash Attention: 通过 IO 感知算法加速注意力计算
        """
    )


if __name__ == "__main__":
    # 测试运行
    attention_analysis_tab()
