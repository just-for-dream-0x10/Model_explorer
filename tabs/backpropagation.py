"""
反向传播原理深度解析模块
Backpropagation Deep Dive Module
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from simple_latex import display_latex


def backpropagation_tab(CHINESE_SUPPORTED):
    """反向传播标签页内容"""

    st.header("🔬 反向传播原理深度解析")

    # ==========================================
    # 第一部分：核心概念介绍
    # ==========================================
    with st.expander("💡 反向传播核心概念", expanded=True):
        st.markdown(
            """
        **反向传播算法 (Backpropagation)** 是训练神经网络的核心算法，基于链式法则高效计算梯度。
        
        **核心思想**：
        1. 🔄 **前向传播** - 计算网络输出和损失
        2. ⬅️ **反向传播** - 从输出层向输入层逐层计算梯度
        3. 🔗 **链式法则** - 将复杂的梯度计算分解为简单的局部梯度乘积
        4. 📉 **梯度下降** - 使用梯度更新参数
        
        **数学基础**：
        """
        )

        display_latex(
            r"\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_i}"
        )

        st.markdown(
            """
        其中：
        - $L$: 损失函数
        - $y$: 输出
        - $z$: 激活前的值
        - $w_i$: 权重参数
        """
        )

    # ==========================================
    # 第二部分：网络类型选择
    # ==========================================
    st.markdown("---")
    network_type = st.selectbox(
        "🎯 选择网络类型进行深度分析",
        ["简单全连接网络", "CNN卷积网络", "RNN循环网络"],
        key="bp_network_type",
    )

    if network_type == "简单全连接网络":
        _fcn_backprop(CHINESE_SUPPORTED)
    elif network_type == "CNN卷积网络":
        _cnn_backprop(CHINESE_SUPPORTED)
    else:  # RNN循环网络
        _rnn_backprop(CHINESE_SUPPORTED)


def _fcn_backprop(CHINESE_SUPPORTED):
    """全连接网络反向传播"""
    st.markdown("### 🔗 全连接网络反向传播")

    # 网络结构参数
    col1, col2, col3 = st.columns(3)
    with col1:
        input_dim = st.slider("输入维度", 2, 5, 3, key="fcn_input_dim")
    with col2:
        hidden_dim = st.slider("隐藏层维度", 2, 6, 4, key="fcn_hidden_dim")
    with col3:
        output_dim = st.slider("输出维度", 2, 4, 2, key="fcn_output_dim")

    # 随机初始化参数
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros(output_dim)

    # 样本数据
    x = np.random.randn(input_dim)
    y_true = np.zeros(output_dim)
    y_true[0] = 1  # one-hot编码

    # 前向传播计算
    st.markdown("---")
    st.markdown("### 📊 前向传播过程")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 网络参数")
        st.markdown(f"- 输入维度: **{input_dim}**")
        st.markdown(f"- 隐藏层维度: **{hidden_dim}**")
        st.markdown(f"- 输出维度: **{output_dim}**")

        st.markdown("**权重矩阵 W1** (输入→隐藏):")
        st.dataframe(pd.DataFrame(W1.round(3)), width=300)

        st.markdown("**权重矩阵 W2** (隐藏→输出):")
        st.dataframe(pd.DataFrame(W2.round(3)), width=300)

    with col2:
        st.markdown("#### 前向计算")

        # 第一层
        z1 = W1.T @ x + b1
        a1 = np.maximum(0, z1)  # ReLU激活

        st.markdown("**隐藏层计算:**")
        display_latex(r"z_1 = W_1^T x + b_1")
        display_latex(r"a_1 = \text{ReLU}(z_1) = \max(0, z_1)")

        df_hidden = pd.DataFrame({"z1": z1.round(3), "a1 (ReLU)": a1.round(3)})
        st.dataframe(df_hidden, width=300)

        # 第二层
        z2 = W2.T @ a1 + b2
        a2 = np.exp(z2) / np.sum(np.exp(z2))  # Softmax

        st.markdown("**输出层计算:**")
        display_latex(r"z_2 = W_2^T a_1 + b_2")
        display_latex(r"a_2 = \text{Softmax}(z_2)")

        df_output = pd.DataFrame(
            {"z2": z2.round(3), "a2 (Softmax)": a2.round(3), "y_true": y_true}
        )
        st.dataframe(df_output, width=400)

        # 损失计算
        loss = -np.sum(y_true * np.log(a2 + 1e-10))
        st.metric("交叉熵损失 (Cross-Entropy Loss)", f"{loss:.4f}")

    # 反向传播计算
    st.markdown("---")
    st.markdown("### ⬅️ 反向传播过程")

    st.markdown(
        """
    **反向传播使用链式法则逐层计算梯度：**
    """
    )

    # 输出层梯度
    dz2 = a2 - y_true
    dW2 = np.outer(a1, dz2)
    db2 = dz2

    # 隐藏层梯度
    da1 = W2 @ dz2
    dz1 = da1 * (z1 > 0).astype(float)  # ReLU导数
    dW1 = np.outer(x, dz1)
    db1 = dz1

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📤 输出层梯度")

        st.markdown("**步骤1: 损失对 z2 的梯度**")
        display_latex(r"\frac{\partial L}{\partial z_2} = a_2 - y_{true}")
        st.dataframe(pd.DataFrame({"dz2": dz2.round(3)}), width=200)

        st.markdown("**步骤2: 损失对 W2 的梯度**")
        display_latex(r"\frac{\partial L}{\partial W_2} = a_1 \otimes dz_2")
        st.dataframe(pd.DataFrame(dW2.round(3)), width=300)

        st.markdown("**步骤3: 损失对 b2 的梯度**")
        display_latex(r"\frac{\partial L}{\partial b_2} = dz_2")
        st.dataframe(pd.DataFrame({"db2": db2.round(3)}), width=200)

    with col2:
        st.markdown("#### 📥 隐藏层梯度")

        st.markdown("**步骤1: 损失对 a1 的梯度**")
        display_latex(r"\frac{\partial L}{\partial a_1} = W_2 \cdot dz_2")
        st.dataframe(pd.DataFrame({"da1": da1.round(3)}), width=200)

        st.markdown("**步骤2: 损失对 z1 的梯度**")
        display_latex(r"\frac{\partial L}{\partial z_1} = da_1 \odot \text{ReLU}'(z_1)")
        st.markdown("*(ReLU导数: 当 z1 > 0 时为 1，否则为 0)*")
        st.dataframe(pd.DataFrame({"dz1": dz1.round(3)}), width=200)

        st.markdown("**步骤3: 损失对 W1 的梯度**")
        display_latex(r"\frac{\partial L}{\partial W_1} = x \otimes dz_1")
        st.dataframe(pd.DataFrame(dW1.round(3)), width=300)

        st.markdown("**步骤4: 损失对 b1 的梯度**")
        display_latex(r"\frac{\partial L}{\partial b_1} = dz_1")
        st.dataframe(pd.DataFrame({"db1": db1.round(3)}), width=200)

    # 梯度验证
    st.markdown("---")
    st.markdown("### ✅ 梯度验证 (Gradient Checking)")

    st.markdown(
        """
    **数值梯度法验证解析梯度的正确性：**
    """
    )

    display_latex(
        r"\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}"
    )

    def compute_loss(x, y_true, W1, b1, W2, b2):
        z1 = W1.T @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2.T @ a1 + b2
        a2 = np.exp(z2) / np.sum(np.exp(z2))
        return -np.sum(y_true * np.log(a2 + 1e-10))

    # 数值梯度计算（只计算W2的一部分以节省时间）
    epsilon = 1e-5
    numerical_dW2 = np.zeros_like(W2)

    for i in range(min(W2.shape[0], 3)):  # 只计算前3行
        for j in range(W2.shape[1]):
            W2_plus = W2.copy()
            W2_minus = W2.copy()
            W2_plus[i, j] += epsilon
            W2_minus[i, j] -= epsilon

            loss_plus = compute_loss(x, y_true, W1, b1, W2_plus, b2)
            loss_minus = compute_loss(x, y_true, W1, b1, W2_minus, b2)

            numerical_dW2[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # 比较解析梯度和数值梯度
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**解析梯度 (Analytical)**")
        st.dataframe(pd.DataFrame(dW2[:3].round(6)), width=250)

    with col2:
        st.markdown("**数值梯度 (Numerical)**")
        st.dataframe(pd.DataFrame(numerical_dW2[:3].round(6)), width=250)

    with col3:
        st.markdown("**差异**")
        diff_matrix = np.abs(dW2[:3] - numerical_dW2[:3])
        st.dataframe(pd.DataFrame(diff_matrix.round(8)), width=250)

    diff = np.mean(np.abs(dW2[:3] - numerical_dW2[:3]))

    if diff < 1e-7:
        st.success(f"✅ 梯度计算正确！平均差异: {diff:.2e}")
    elif diff < 1e-5:
        st.warning(f"⚠️ 梯度计算可能有小误差。平均差异: {diff:.2e}")
    else:
        st.error(f"❌ 梯度计算可能有误！平均差异: {diff:.2e}")

    # 梯度流可视化
    st.markdown("---")
    st.markdown("### 📊 梯度流可视化")

    # 计算每层的梯度范数
    grad_norms = {
        "dW2": np.linalg.norm(dW2),
        "db2": np.linalg.norm(db2),
        "dW1": np.linalg.norm(dW1),
        "db1": np.linalg.norm(db1),
    }

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(grad_norms.keys()),
                y=list(grad_norms.values()),
                text=[f"{v:.4f}" for v in grad_norms.values()],
                textposition="auto",
                marker_color=["#FF6B6B", "#FFA07A", "#4ECDC4", "#95E1D3"],
            )
        ]
    )

    fig.update_layout(
        title="各层梯度的L2范数", xaxis_title="参数", yaxis_title="梯度范数", height=400
    )

    st.plotly_chart(fig, width="stretch")


def _cnn_backprop(CHINESE_SUPPORTED):
    """CNN卷积网络反向传播"""
    st.markdown("### 🖼️ CNN卷积网络反向传播")

    st.markdown(
        """
    **卷积层反向传播的关键：**
    - 梯度对输入的传播需要旋转卷积核
    - 梯度对卷积核的计算类似于前向卷积操作
    """
    )

    # 数学公式
    st.markdown("#### 📐 卷积层反向传播公式")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**损失对卷积核的梯度：**")
        display_latex(
            r"\frac{\partial L}{\partial K} = X * \frac{\partial L}{\partial Y}"
        )
        st.markdown("其中 $*$ 表示卷积运算")

    with col2:
        st.markdown("**损失对输入的梯度：**")
        display_latex(
            r"\frac{\partial L}{\partial X} = \text{rot}_{180}(K) * \frac{\partial L}{\partial Y}"
        )
        st.markdown("其中 $\\text{rot}_{180}$ 表示180度旋转")

    # 实际计算示例
    st.markdown("---")
    st.markdown("### 🧮 卷积反向传播计算示例")

    # 参数设置
    col1, col2 = st.columns(2)
    with col1:
        input_size = st.slider("输入尺寸", 4, 8, 5, key="cnn_input_size")
    with col2:
        kernel_size = st.slider("卷积核尺寸", 2, 4, 3, key="cnn_kernel_size")

    # 生成数据
    np.random.seed(42)
    x = np.random.randn(input_size, input_size)
    K = np.random.randn(kernel_size, kernel_size)

    # 前向卷积
    y = signal.convolve2d(x, K, mode="valid")
    output_size = y.shape[0]

    # 假设损失对输出的梯度（简化为全1）
    dL_dy = np.ones_like(y)

    # 反向传播
    dL_dK = signal.convolve2d(x, dL_dy, mode="valid")
    dL_dx = signal.convolve2d(dL_dy, np.rot90(K, 2), mode="full")

    # 显示结果
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📊 前向传播")

        st.markdown(f"**输入 X** ({input_size}×{input_size}):")
        st.dataframe(pd.DataFrame(x.round(3)), width=350)

        st.markdown(f"**卷积核 K** ({kernel_size}×{kernel_size}):")
        st.dataframe(pd.DataFrame(K.round(3)), width=250)

        st.markdown(f"**输出 Y** ({output_size}×{output_size}):")
        st.dataframe(pd.DataFrame(y.round(3)), width=250)

    with col2:
        st.markdown("#### ⬅️ 反向传播")

        st.markdown(
            f"**损失对输出的梯度** $\\frac{{\\partial L}}{{\\partial Y}}$ ({output_size}×{output_size}):"
        )
        st.dataframe(pd.DataFrame(dL_dy.round(3)), width=250)

        st.markdown(
            f"**损失对卷积核的梯度** $\\frac{{\\partial L}}{{\\partial K}}$ ({kernel_size}×{kernel_size}):"
        )
        st.dataframe(pd.DataFrame(dL_dK.round(3)), width=250)

        st.markdown(
            f"**损失对输入的梯度** $\\frac{{\\partial L}}{{\\partial X}}$ ({input_size}×{input_size}):"
        )
        st.dataframe(pd.DataFrame(dL_dx.round(3)), width=350)

    # 可视化
    st.markdown("---")
    st.markdown("### 📊 梯度热力图可视化")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("输入梯度 dL/dX", "卷积核梯度 dL/dK", "输出梯度 dL/dY"),
    )

    fig.add_trace(go.Heatmap(z=dL_dx, colorscale="RdBu", zmid=0), row=1, col=1)

    fig.add_trace(go.Heatmap(z=dL_dK, colorscale="RdBu", zmid=0), row=1, col=2)

    fig.add_trace(go.Heatmap(z=dL_dy, colorscale="RdBu", zmid=0), row=1, col=3)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def _rnn_backprop(CHINESE_SUPPORTED):
    """RNN循环网络反向传播"""
    st.markdown("### 🔄 RNN循环网络反向传播 (BPTT)")

    st.markdown(
        """
    **Backpropagation Through Time (BPTT)** 是RNN的反向传播算法，梯度需要通过时间反向传播。
    
    **关键挑战：**
    - 梯度消失：长序列导致梯度指数级衰减
    - 梯度爆炸：权重过大导致梯度指数级增长
    """
    )

    # 数学公式
    st.markdown("#### 📐 RNN反向传播公式")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**前向传播：**")
        display_latex(r"h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)")
        display_latex(r"y_t = W_{hy}h_t + b_y")

    with col2:
        st.markdown("**反向传播：**")
        display_latex(
            r"\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}"
        )
        st.markdown("梯度通过时间反向传播")

    # 实际计算示例
    st.markdown("---")
    st.markdown("### 🧮 RNN反向传播计算示例")

    # 参数设置
    col1, col2, col3 = st.columns(3)
    with col1:
        seq_len = st.slider("序列长度", 2, 5, 3, key="rnn_seq_len")
    with col2:
        input_dim = st.slider("输入维度", 2, 4, 2, key="rnn_input_dim")
    with col3:
        hidden_dim = st.slider("隐藏层维度", 2, 4, 3, key="rnn_hidden_dim")

    # 初始化参数
    np.random.seed(42)
    W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
    W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
    W_hy = np.random.randn(hidden_dim, 1) * 0.1
    b_h = np.zeros(hidden_dim)

    # 输入序列
    X = np.random.randn(seq_len, input_dim)

    # 前向传播
    h = np.zeros((seq_len, hidden_dim))
    z = np.zeros((seq_len, hidden_dim))

    for t in range(seq_len):
        if t == 0:
            z[t] = W_xh.T @ X[t] + b_h
        else:
            z[t] = W_xh.T @ X[t] + W_hh.T @ h[t - 1] + b_h
        h[t] = np.tanh(z[t])

    # 输出
    Y = h @ W_hy

    # 显示前向传播结果
    st.markdown("#### 📊 前向传播过程")

    for t in range(seq_len):
        with st.expander(f"⏱️ 时间步 t={t+1}", expanded=(t == 0)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**输入 x_{t}:**")
                st.dataframe(pd.DataFrame(X[t].reshape(1, -1).round(3)), width=200)

            with col2:
                st.markdown(f"**隐藏状态 h_{t}:**")
                st.dataframe(pd.DataFrame(h[t].reshape(1, -1).round(3)), width=200)

            with col3:
                st.markdown(f"**输出 y_{t}:**")
                st.dataframe(pd.DataFrame(Y[t].reshape(1, -1).round(3)), width=150)

    # 梯度消失/爆炸演示
    st.markdown("---")
    st.markdown("### ⚠️ 梯度消失与梯度爆炸演示")

    st.markdown(
        """
    **梯度在时间上的传播：**
    
    梯度在反向传播过程中会连续乘以权重矩阵的转置。如果权重的特征值不接近1，就会出现问题。
    """
    )

    # 模拟梯度传播
    col1, col2 = st.columns(2)

    with col1:
        num_steps = st.slider("反向传播时间步数", 5, 50, 20, key="rnn_bptt_steps")
    with col2:
        weight_eigenvalue = st.slider(
            "权重矩阵特征值", 0.3, 2.0, 1.0, step=0.1, key="rnn_eigenvalue"
        )

    # 计算梯度传播
    gradients = []
    grad = 1.0
    for t in range(num_steps):
        grad = grad * weight_eigenvalue
        gradients.append(grad)

    # 可视化
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(num_steps)),
            y=gradients,
            mode="lines+markers",
            name=f"特征值={weight_eigenvalue}",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=f"梯度传播演示 (特征值={weight_eigenvalue})",
        xaxis_title="反向传播时间步",
        yaxis_title="梯度大小",
        yaxis_type="log",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    # 分析结果
    final_grad = gradients[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("初始梯度", "1.0")

    with col2:
        st.metric("最终梯度", f"{final_grad:.2e}")

    with col3:
        ratio = final_grad / 1.0
        st.metric("梯度变化率", f"{ratio:.2e}")

    # 诊断
    if weight_eigenvalue > 1.1:
        st.error(f"⚠️ **梯度爆炸风险！** 特征值 > 1，梯度从 1.0 增长到 {final_grad:.2e}")
        st.markdown("**解决方案：** 梯度裁剪 (Gradient Clipping)、权重正则化")
    elif weight_eigenvalue < 0.9:
        st.warning(
            f"⚠️ **梯度消失风险！** 特征值 < 1，梯度从 1.0 衰减到 {final_grad:.2e}"
        )
        st.markdown("**解决方案：** 使用LSTM/GRU、残差连接")
    else:
        st.success(f"✅ **梯度稳定！** 特征值 ≈ 1，梯度保持相对稳定")


if __name__ == "__main__":
    # 独立运行时的测试
    backpropagation_tab(True)
