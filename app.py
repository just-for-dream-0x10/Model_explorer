import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from sympy import symbols, Matrix, simplify, latex
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
import locale
import sys
import platform
from simple_latex import display_latex, display_formula_box, display_math_content


# ==========================================
# 全局中文字体配置
# ==========================================


def detect_chinese_support():
    """检测系统是否支持中文显示"""
    try:
        # 检测系统语言环境
        system_language = locale.getdefaultlocale()[0]
        if system_language and "zh" in system_language.lower():
            return True

        # 检测系统编码
        if sys.getdefaultencoding().lower().startswith("utf"):
            return True

        # 尝试显示中文字符
        test_str = "测试"
        test_str.encode(sys.getdefaultencoding())
        return True
    except:
        return False


def configure_matplotlib_font():
    """配置matplotlib字体以支持中文"""
    chinese_supported = detect_chinese_support()

    if chinese_supported:
        try:
            # 尝试设置中文字体
            if platform.system() == "Darwin":  # macOS
                plt.rcParams["font.sans-serif"] = [
                    "Arial Unicode MS",
                    "PingFang SC",
                    "SimHei",
                    "Microsoft YaHei",
                ]
            elif platform.system() == "Windows":
                plt.rcParams["font.sans-serif"] = [
                    "SimHei",
                    "Microsoft YaHei",
                    "Arial Unicode MS",
                ]
            else:  # Linux
                plt.rcParams["font.sans-serif"] = [
                    "DejaVu Sans",
                    "SimHei",
                    "Arial Unicode MS",
                ]

            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "测试", fontsize=12)
            plt.close(fig)
            return True

        except:
            # 如果中文字体设置失败，使用英文
            plt.rcParams["font.sans-serif"] = [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
            ]
            return False
    else:
        # 系统不支持中文，使用英文字体
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
        return False


# 执行字体配置
CHINESE_SUPPORTED = configure_matplotlib_font()

# 中英文文本配置
TEXT_CONFIG = {
    "chinese": {
        "title": "神经网络数学原理探索器",
        "subtitle": "深入理解CNN、GNN、RNN等神经网络的核心数学原理",
        "description": "**交互式数学实验室** - 通过严谨的数学推导和可视化深入理解神经网络算法",
        "cnn_tab": "🔄 CNN卷积数学",
        "gnn_tab": "🕸️ GNN图神经网络",
        "rnn_tab": "🔄 RNN/LSTM时序网络",
        "math_tab": "📐 数学推导工具",
        "bp_tab": "🔬 反向传播原理",
        "lab_tab": "🎮 交互实验室",
        "params_title": "🎛️ 实验参数",
        "cnn_params": "CNN 参数",
        "gnn_params": "GNN 参数",
        "rnn_params": "RNN/LSTM 参数",
        "common_params": "通用参数",
        "learning_rate": "学习率",
    },
    "english": {
        "title": "Neural Network Mathematics Explorer",
        "subtitle": "Deep Understanding of Core Mathematical Principles in CNN, GNN, RNN and Other Neural Networks",
        "description": "**Interactive Mathematics Lab** - Deep understanding of neural network algorithms through rigorous mathematical derivations and visualizations",
        "cnn_tab": "🔄 CNN Convolution Math",
        "gnn_tab": "🕸️ GNN Graph Neural Networks",
        "rnn_tab": "🔄 RNN/LSTM Sequential Networks",
        "math_tab": "📐 Math Derivation Tools",
        "bp_tab": "🔬 Backpropagation Principles",
        "lab_tab": "🎮 Interactive Lab",
        "params_title": "🎛️ Experiment Parameters",
        "cnn_params": "CNN Parameters",
        "gnn_params": "GNN Parameters",
        "rnn_params": "RNN/LSTM Parameters",
        "common_params": "Common Parameters",
        "learning_rate": "Learning Rate",
    },
}


# 获取当前语言的文本配置
def get_text(key):
    return TEXT_CONFIG["chinese" if CHINESE_SUPPORTED else "english"][key]


# 页面配置
st.set_page_config(
    page_title=get_text("title"),
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS样式和LaTeX配置
st.markdown(
    """
<style>
    .math-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .formula-box {
        background-color: #e3f2fd;
        border: 1px solid #2196F3;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>

<!-- KaTeX CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0KOVEMeaKrumfonJpasSUgnkYtGIYLpAkH5EVWNeDNJg8jVnbYiVT" crossorigin="anonymous">

<!-- KaTeX JS -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzYCPaOoIrBvzqhzd2Fh+R7d4QG4G4G4G4G4" crossorigin="anonymous"></script>

<!-- KaTeX Auto-render -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>

<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\\\[', right: '\\\\]', display: true},
                {left: '\\\\(', right: '\\\\)', display: false}
            ],
            throwOnError: false
        });
    });
</script>
""",
    unsafe_allow_html=True,
)

st.title("🧮 " + get_text("title"))
st.markdown("### " + get_text("subtitle"))
st.markdown(get_text("description"))

# 侧边栏参数控制
with st.sidebar:
    st.header(get_text("params_title"))

    st.subheader(get_text("cnn_params"))
    kernel_size = st.slider(
        "卷积核大小" if CHINESE_SUPPORTED else "Kernel Size", 1, 7, 3
    )
    stride = st.slider("步长" if CHINESE_SUPPORTED else "Stride", 1, 4, 1)
    padding = st.slider("填充" if CHINESE_SUPPORTED else "Padding", 0, 3, 0)

    st.subheader(get_text("gnn_params"))
    num_nodes = st.slider(
        "节点数量" if CHINESE_SUPPORTED else "Number of Nodes", 3, 10, 5
    )
    num_layers = st.slider("GNN层数" if CHINESE_SUPPORTED else "GNN Layers", 1, 5, 2)

    st.subheader(get_text("rnn_params"))
    sequence_length = st.slider(
        "序列长度" if CHINESE_SUPPORTED else "Sequence Length", 5, 50, 20
    )
    hidden_size = st.slider(
        "隐藏层大小" if CHINESE_SUPPORTED else "Hidden Size", 4, 64, 16
    )
    rnn_type = st.selectbox(
        "RNN类型" if CHINESE_SUPPORTED else "RNN Type", ["Simple RNN", "LSTM", "GRU"]
    )

    st.subheader(get_text("common_params"))
    learning_rate = st.select_slider(
        get_text("learning_rate"),
        options=[1e-4, 1e-3, 1e-2, 1e-1],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}",
    )

# 主界面标签页
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        get_text("cnn_tab"),
        get_text("gnn_tab"),
        get_text("rnn_tab"),
        "🌊 扩散模型" if CHINESE_SUPPORTED else "🌊 Diffusion Models",
        get_text("math_tab"),
        get_text("bp_tab"),
        get_text("lab_tab"),
    ]
)

# 导入模块
from cnn import cnn_tab
from gnn import gnn_tab
from rnn_lstm import rnn_lstm_tab
from diffusion import diffusion_tab

# TAB 1: CNN卷积数学
with tab1:
    cnn_tab(CHINESE_SUPPORTED, kernel_size, stride, padding)

# TAB 2: GNN图神经网络
with tab2:
    gnn_tab(CHINESE_SUPPORTED, num_nodes, num_layers)

# TAB 3: RNN/LSTM时序网络
with tab3:
    rnn_lstm_tab(CHINESE_SUPPORTED)

# TAB 4: 扩散模型
with tab4:
    diffusion_tab(CHINESE_SUPPORTED)

# TAB 5: 数学推导工具
with tab5:
    st.header("📐 交互式数学推导工具")

    derivation_type = st.selectbox(
        "选择推导主题",
        ["卷积定理推导", "梯度下降优化", "反向传播链式法则", "图拉普拉斯矩阵"],
    )

    if derivation_type == "卷积定理推导":
        st.markdown("### 卷积定理数学推导")

        st.markdown("#### 定理陈述：")
        display_latex(
            "\\mathcal{F}\\{f * g\\} = \\mathcal{F}\\{f\\} \\cdot \\mathcal{F}\\{g\\}"
        )

        st.markdown("#### 证明：")
        display_latex(
            "\\mathcal{F}\\{f * g\\}(\\omega) = \\int (f * g)(t) e^{-i\\omega t} dt"
        )
        display_latex("= \\iint f(\\tau)g(t-\\tau) e^{-i\\omega t} d\\tau dt")
        st.markdown("令 $u = t-\\tau$，则 $t = u+\\tau$，$dt = du$")
        display_latex("= \\iint f(\\tau)g(u) e^{-i\\omega(u+\\tau)} d\\tau du")
        display_latex(
            "= \\int f(\\tau) e^{-i\\omega \\tau} d\\tau \\cdot \\int g(u) e^{-i\\omega u} du"
        )
        display_latex(
            "= \\mathcal{F}\\{f\\}(\\omega) \\cdot \\mathcal{F}\\{g\\}(\\omega)"
        )
        st.markdown("$\\square$ 证毕")

        st.markdown("### 数值验证")

        # 创建信号
        t = np.linspace(0, 1, 100)
        f = np.sin(2 * np.pi * 5 * t)  # 5Hz正弦波
        g = np.exp(-10 * (t - 0.5) ** 2)  # 高斯窗

        # 时域卷积
        conv_time = np.convolve(f, g, mode="same")

        # 频域乘积
        F_f = np.fft.fft(f)
        F_g = np.fft.fft(g)
        conv_freq = np.fft.ifft(F_f * F_g).real

        # 可视化
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=t, y=conv_time, name="时域卷积", line=dict(color="blue"))
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=conv_freq, name="频域乘积", line=dict(color="red", dash="dash")
            )
        )
        fig.update_layout(title="卷积定理验证", xaxis_title="时间", yaxis_title="幅度")
        st.plotly_chart(fig, width="stretch")

        # 计算误差
        error = np.mean(np.abs(conv_time - conv_freq))
        st.metric("数值误差", f"{error:.2e}")

    elif derivation_type == "梯度下降优化":
        st.markdown("### 梯度下降数学推导")

        st.markdown("#### 目标函数：")
        display_latex(
            "J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2"
        )

        st.markdown("#### 梯度计算：")
        display_latex(
            "\\frac{\\partial J}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}"
        )

        st.markdown("#### 更新规则：")
        display_latex(
            "\\theta_j := \\theta_j - \\alpha \\frac{\\partial J}{\\partial \\theta_j}"
        )
        st.markdown("其中 $\\alpha$ 是学习率")

        # 交互式梯度下降可视化
        st.markdown("### 梯度下降可视化")

        # 创建二次函数
        x = np.linspace(-5, 5, 100)
        y = x**2

        # 梯度下降模拟
        start_x = st.slider("起始点", -4.0, 4.0, 3.0)
        lr = st.slider("学习率", 0.01, 0.5, 0.1)
        iterations = st.slider("迭代次数", 10, 100, 50)

        # 执行梯度下降
        path_x = [start_x]
        path_y = [start_x**2]

        current_x = start_x
        for i in range(iterations):
            grad = 2 * current_x  # f'(x) = 2x
            current_x = current_x - lr * grad
            path_x.append(current_x)
            path_y.append(current_x**2)

        # 可视化
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x, y=y, name="f(x) = x²", line=dict(color="lightgray"))
        )
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                name="梯度下降路径",
                mode="markers+lines",
                line=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[path_x[-1]],
                y=[path_y[-1]],
                name="收敛点",
                mode="markers",
                marker=dict(size=10, color="green"),
            )
        )
        fig.update_layout(
            title=f"梯度下降 (α={lr}, 迭代={iterations})",
            xaxis_title="x",
            yaxis_title="f(x)",
        )
        st.plotly_chart(fig, width="stretch")

        st.metric("最终位置", f"x = {path_x[-1]:.4f}")
        st.metric("最终函数值", f"f(x) = {path_y[-1]:.4f}")

    elif derivation_type == "反向传播链式法则":
        st.markdown("### 反向传播链式法则推导")

        st.markdown("#### 链式法则：")
        st.markdown("对于复合函数 $y = f(g(x))$，有：")
        display_latex("\\frac{dy}{dx} = \\frac{dy}{dg} \cdot \\frac{dg}{dx}")

        st.markdown("#### 神经网络中的应用：")
        st.markdown("对于 $L$ 层网络，损失函数对第 $l$ 层参数的梯度：")
        display_latex(
            "\\frac{\\partial L}{\\partial W^{(l)}} = \\frac{\\partial L}{\\partial a^{(L)}} \cdot \\frac{\\partial a^{(L)}}{\\partial z^{(L)}} \cdot \ldots \cdot \\frac{\\partial a^{(l)}}{\\partial z^{(l)}} \cdot \\frac{\\partial z^{(l)}}{\\partial W^{(l)}}"
        )
        st.markdown("其中：")
        st.markdown("- $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$")
        st.markdown("- $a^{(l)} = \\sigma(z^{(l)})$")

        # 简单网络示例
        st.markdown("### 简单网络反向传播示例")

        # 2-1网络
        x1, x2 = symbols("x1 x2")
        w1, w2, b = symbols("w1 w2 b")

        # 前向传播
        z = w1 * x1 + w2 * x2 + b
        a = z  # 线性激活
        L = a**2  # 简单损失函数

        st.markdown("#### 网络结构：")
        st.markdown("- 输入：x₁, x₂")
        st.markdown("- 权重：w₁, w₂")
        st.markdown("- 偏置：b")
        st.markdown("- 输出：a = w₁x₁ + w₂x₂ + b")
        st.markdown("- 损失：L = a²")

        # 计算梯度
        dL_dw1 = L.diff(w1)
        dL_dw2 = L.diff(w2)
        dL_db = L.diff(b)

        st.markdown("#### 梯度计算：")
        st.markdown(f"$\\frac{{\\partial L}}{{\\partial w_1}} = {latex(dL_dw1)}$")
        st.markdown(f"$\\frac{{\\partial L}}{{\\partial w_2}} = {latex(dL_dw2)}$")
        st.markdown(f"$\\frac{{\\partial L}}{{\\partial b}} = {latex(dL_db)}$")

        # 数值验证
        st.markdown("#### 数值验证：")
        col1, col2 = st.columns([1, 1])

        with col1:
            # 设置参数值
            x1_val = st.number_input("x₁", value=1.0)
            x2_val = st.number_input("x₂", value=2.0)
            w1_val = st.number_input("w₁", value=0.5)
            w2_val = st.number_input("w₂", value=-0.3)
            b_val = st.number_input("b", value=0.1)

        with col2:
            # 计算数值梯度
            z_val = w1_val * x1_val + w2_val * x2_val + b_val
            a_val = z_val
            L_val = a_val**2

            dL_dw1_val = 2 * a_val * x1_val
            dL_dw2_val = 2 * a_val * x2_val
            dL_db_val = 2 * a_val

            st.markdown(f"前向传播：z = {z_val:.3f}, a = {a_val:.3f}, L = {L_val:.3f}")
            st.markdown(f"$\\frac{{\\partial L}}{{\\partial w_1}} = {dL_dw1_val:.3f}$")
            st.markdown(f"$\\frac{{\\partial L}}{{\\partial w_2}} = {dL_dw2_val:.3f}$")
            st.markdown(f"$\\frac{{\\partial L}}{{\\partial b}} = {dL_db_val:.3f}$")

    elif derivation_type == "图拉普拉斯矩阵":
        st.markdown("### 图拉普拉斯矩阵数学推导")

        st.markdown("#### 定义：")
        display_latex("L = D - A")
        st.markdown("其中：")
        st.markdown("- $D$ 是度矩阵（对角矩阵，$D_{ii} = $ 节点$i$的度）")
        st.markdown("- $A$ 是邻接矩阵")

        st.markdown("#### 归一化拉普拉斯矩阵：")
        display_latex("L_{sym} = I - D^{-1/2}AD^{-1/2}")

        st.markdown("#### 性质：")
        st.markdown("1. $L$ 是半正定矩阵")
        st.markdown("2. 特征值都是非负的")
        st.markdown("3. 最小特征值为0，对应的特征向量为全1向量")

        # 创建示例图并计算拉普拉斯矩阵
        st.markdown("### 拉普拉斯矩阵计算示例")

        # 创建示例图
        num_nodes = 5  # 使用5个节点的示例图
        G = nx.erdos_renyi_graph(num_nodes, 0.4, seed=42)
        A = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(A, axis=1))
        L = D - A

        # 归一化拉普拉斯
        try:
            D_inv_sqrt = np.linalg.inv(np.sqrt(D))
            L_sym = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
        except np.linalg.LinAlgError:
            # 处理奇异矩阵情况（度为0的节点）
            D_sqrt = np.sqrt(D)
            D_inv_sqrt = np.zeros_like(D_sqrt)
            # 只对非零元素求逆
            non_zero_mask = D_sqrt > 1e-10
            D_inv_sqrt[non_zero_mask] = 1.0 / D_sqrt[non_zero_mask]
            L_sym = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 度矩阵 D")
            st.dataframe(
                pd.DataFrame(
                    D,
                    index=[f"Node {i}" for i in range(num_nodes)],
                    columns=[f"Node {i}" for i in range(num_nodes)],
                )
            )

            st.markdown("#### 拉普拉斯矩阵 L = D - A")
            st.dataframe(
                pd.DataFrame(
                    L,
                    index=[f"Node {i}" for i in range(num_nodes)],
                    columns=[f"Node {i}" for i in range(num_nodes)],
                )
            )

        with col2:
            st.markdown("#### 归一化拉普拉斯 $L_{sym}$")
            st.dataframe(
                pd.DataFrame(
                    L_sym.round(3),
                    index=[f"Node {i}" for i in range(num_nodes)],
                    columns=[f"Node {i}" for i in range(num_nodes)],
                )
            )

            # 特征值分解
            eigenvals, eigenvecs = np.linalg.eigh(L_sym)
            st.markdown("#### 特征值")
            eigen_df = pd.DataFrame(
                {"特征值": eigenvals.round(4), "索引": range(len(eigenvals))}
            )
            st.dataframe(eigen_df)

            # 特征值可视化
            fig = px.bar(
                x=range(len(eigenvals)),
                y=eigenvals,
                labels={"x": "特征值索引", "y": "特征值"},
                title="拉普拉斯矩阵特征值谱",
            )
            st.plotly_chart(fig, width="stretch")

# TAB 6: 反向传播原理
with tab6:
    st.header("🔬 反向传播原理深度解析")

    st.markdown("### 反向传播的核心思想")
    st.markdown(
        """
    反向传播算法是训练神经网络的核心，基于链式法则高效计算梯度。
    """
    )

    network_type = st.selectbox(
        "选择网络类型", ["简单全连接网络", "CNN卷积网络", "RNN循环网络", "扩散模型 (Diffusion)"]
    )

    if network_type == "简单全连接网络":
        st.markdown("### 全连接网络反向传播")

        # 创建简单的2层网络
        input_dim = 3
        hidden_dim = 4
        output_dim = 2

        # 随机初始化参数
        np.random.seed(42)
        W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        b2 = np.zeros(output_dim)

        # 样本数据
        x = np.random.randn(input_dim)
        y_true = np.array([1, 0])  # one-hot编码

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 网络参数")
            st.markdown(f"输入维度: {input_dim}")
            st.markdown(f"隐藏层维度: {hidden_dim}")
            st.markdown(f"输出维度: {output_dim}")

            st.markdown("W1 (输入→隐藏):")
            st.dataframe(pd.DataFrame(W1.round(3)))

            st.markdown("W2 (隐藏→输出):")
            st.dataframe(pd.DataFrame(W2.round(3)))

        with col2:
            st.markdown("#### 前向传播")

            # 第一层
            z1 = W1.T @ x + b1
            a1 = np.maximum(0, z1)  # ReLU激活

            # 第二层
            z2 = W2.T @ a1 + b2
            a2 = np.exp(z2) / np.sum(np.exp(z2))  # Softmax

            st.markdown("隐藏层激活:")
            st.dataframe(pd.DataFrame({"z1": z1.round(3), "a1": a1.round(3)}))

            st.markdown("输出层:")
            st.dataframe(pd.DataFrame({"z2": z2.round(3), "a2": a2.round(3)}))

            # 损失计算
            loss = -np.sum(y_true * np.log(a2))
            st.metric("交叉熵损失", f"{loss:.4f}")

        # 反向传播
        st.markdown("#### 反向传播计算")

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
            st.markdown("输出层梯度:")
            st.markdown("$\\frac{{\\partial L}}{{\\partial z_2}} = a_2 - y$")
            st.dataframe(pd.DataFrame({"dz2": dz2.round(3)}))

            st.markdown("$\\frac{{\\partial L}}{{\\partial W_2}} = a_1 \\otimes dz_2$")
            st.dataframe(pd.DataFrame(dW2.round(3)))

            st.markdown("$\\frac{{\\partial L}}{{\\partial b_2}} = dz_2$")
            st.dataframe(pd.DataFrame({"db2": db2.round(3)}))

        with col2:
            st.markdown("隐藏层梯度:")
            st.markdown("$\\frac{{\\partial L}}{{\\partial a_1}} = W_2 \\cdot dz_2$")
            st.dataframe(pd.DataFrame({"da1": da1.round(3)}))

            st.markdown(
                "$\\frac{{\\partial L}}{{\\partial z_1}} = da_1 \\odot \\text{{ReLU}}'(z_1)$"
            )
            st.dataframe(pd.DataFrame({"dz1": dz1.round(3)}))

            st.markdown("$\\frac{{\\partial L}}{{\\partial W_1}} = x \\otimes dz_1$")
            st.dataframe(pd.DataFrame(dW1.round(3)))

            st.markdown("$\\frac{{\\partial L}}{{\\partial b_1}} = dz_1$")
            st.dataframe(pd.DataFrame({"db1": db1.round(3)}))

        # 梯度验证
        st.markdown("#### 数值梯度验证")

        def compute_loss(x, y_true, W1, b1, W2, b2):
            z1 = W1.T @ x + b1
            a1 = np.maximum(0, z1)
            z2 = W2.T @ a1 + b2
            a2 = np.exp(z2) / np.sum(np.exp(z2))
            return -np.sum(y_true * np.log(a2))

        # 数值梯度计算
        epsilon = 1e-5
        numerical_dW2 = np.zeros_like(W2)

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_plus = W2.copy()
                W2_minus = W2.copy()
                W2_plus[i, j] += epsilon
                W2_minus[i, j] -= epsilon

                loss_plus = compute_loss(x, y_true, W1, b1, W2_plus, b2)
                loss_minus = compute_loss(x, y_true, W1, b1, W2_minus, b2)

                numerical_dW2[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # 比较解析梯度和数值梯度
        diff = np.mean(np.abs(dW2 - numerical_dW2))
        st.metric("梯度差异", f"{diff:.2e}")

        if diff < 1e-7:
            st.success("✅ 梯度计算正确！")
        elif diff < 1e-5:
            st.warning("⚠️ 梯度计算可能有小误差")
        else:
            st.error("❌ 梯度计算可能有误")

    elif network_type == "CNN卷积网络":
        st.markdown("### CNN反向传播原理")

        st.markdown("#### 卷积层梯度：")
        display_latex(
            "\\frac{\\partial L}{\\partial K} = \\frac{\\partial L}{\\partial Y} * X_{rotated}"
        )
        display_latex(
            "\\frac{\\partial L}{\\partial X} = K_{rotated} * \\frac{\\partial L}{\\partial Y}"
        )
        st.markdown("其中 $*$ 表示卷积运算，$rotated$ 表示180度旋转")

        # 简单卷积示例
        input_size = 4
        kernel_size = 3
        x = np.random.randn(input_size, input_size)
        K = np.random.randn(kernel_size, kernel_size)

        # 前向卷积
        y = signal.convolve2d(x, K, mode="valid")

        # 假设损失对输出的梯度
        dL_dy = np.ones_like(y)

        # 反向传播
        dL_dK = signal.convolve2d(x, dL_dy, mode="valid")
        dL_dx = signal.convolve2d(dL_dy, np.rot90(K, 2), mode="full")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 前向传播")
            st.markdown("输入 X:")
            st.dataframe(pd.DataFrame(x.round(3)))

            st.markdown("卷积核 K:")
            st.dataframe(pd.DataFrame(K.round(3)))

            st.markdown("输出 Y:")
            st.dataframe(pd.DataFrame(y.round(3)))

        with col2:
            st.markdown("#### 反向传播")
            st.markdown("$\\frac{{\\partial L}}{{\\partial Y}}$:")
            st.dataframe(pd.DataFrame(dL_dy.round(3)))

            st.markdown(
                "$\\frac{{\\partial L}}{{\\partial K}} = X * \\frac{{\\partial L}}{{\\partial Y}}$:"
            )
            st.dataframe(pd.DataFrame(dL_dK.round(3)))

            st.markdown(
                "$\\frac{{\\partial L}}{{\\partial X}} = K_{{rotated}} * \\frac{{\\partial L}}{{\\partial Y}}$"
            )
            st.dataframe(pd.DataFrame(dL_dx.round(3)))

    elif network_type == "RNN循环网络":
        st.markdown("### RNN反向传播 Through Time (BPTT)")

        st.markdown("#### RNN前向传播：")
        display_latex("h_t = \\tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)")
        display_latex("y_t = W_{hy}h_t + b_y")

        st.markdown("#### 反向传播：")
        display_latex(
            "\\frac{\\partial L}{\\partial h_t} = \\frac{\\partial L_t}{\\partial h_t} + \\frac{\\partial L}{\\partial h_{t+1}} \\cdot \\frac{\\partial h_{t+1}}{\\partial h_t}"
        )
        st.markdown("梯度通过时间反向传播")

        # 简单RNN示例
        seq_len = 3
        input_dim = 2
        hidden_dim = 3

        # 参数
        W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
        W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        W_hy = np.random.randn(hidden_dim, 1) * 0.1

        # 输入序列
        X = np.random.randn(seq_len, input_dim)

        # 前向传播
        h = np.zeros((seq_len, hidden_dim))
        for t in range(seq_len):
            if t == 0:
                h[t] = np.tanh(W_xh.T @ X[t])
            else:
                h[t] = np.tanh(W_xh.T @ X[t] + W_hh.T @ h[t - 1])

        # 输出
        Y = h @ W_hy

        st.markdown("#### RNN前向传播")
        for t in range(seq_len):
            st.markdown(f"时间步 {t+1}:")
            st.markdown(
                f"$$ h_{{{t}}} = \\tanh(W_{{xh}} \\cdot x_{{{t}}} + W_{{hh}} \\cdot h_{{{t-1}}}) $$"
            )
            st.dataframe(
                pd.DataFrame(
                    h[t].round(3).reshape(1, -1),
                    columns=[f"h{t}_{i}" for i in range(hidden_dim)],
                )
            )

        st.markdown("#### 梯度消失/爆炸演示")

        # 模拟梯度传播
        num_steps = st.slider("时间步数", 5, 50, 20)
        eigenvalue = st.slider("$W_{hh}$ 特征值", 0.5, 2.0, 1.0)

        gradients = []
        grad = 1.0
        for t in range(num_steps):
            grad = grad * eigenvalue
            gradients.append(grad)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(num_steps)), y=gradients, mode="lines+markers")
        )
        fig.update_layout(
            title=f"梯度传播 (特征值={eigenvalue})",
            xaxis_title="时间步",
            yaxis_title="梯度大小",
            yaxis_type="log",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
        **观察：**
        - 特征值 > 1: 梯度爆炸
        - 特征值 < 1: 梯度消失
        - 特征值 = 1: 梯度保持
        """
        )
    
    elif network_type == "扩散模型 (Diffusion)":
        st.markdown("### 扩散模型反向传播深度解析")
        
        st.markdown("""
        扩散模型的训练目标是学习预测噪声 $\\epsilon_\\theta(x_t, t)$，通过反向传播优化模型参数。
        这里我们深入探讨扩散模型的损失函数、梯度计算和训练过程。
        """)
        
        # 创建子标签
        diff_tab1, diff_tab2, diff_tab3, diff_tab4 = st.tabs([
            "📖 损失函数推导",
            "🔢 梯度计算详解", 
            "🎯 训练目标对比",
            "🧮 数值梯度验证"
        ])
        
        with diff_tab1:
            st.markdown("#### 1️⃣ 变分下界（ELBO）推导")
            
            st.markdown("""
            扩散模型的训练从最大化对数似然开始：
            """)
            
            display_latex(r"\max_\theta \mathbb{E}_{x_0 \sim q(x_0)} [\log p_\theta(x_0)]")
            
            st.markdown("通过变分推断，我们得到变分下界（ELBO）：")
            
            st.markdown("""
            $$
            \\log p_\theta(x_0) \\geq \\mathbb{E}_q \\left[ \\log \\frac{p_\\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\right] = \\mathcal{L}_{\\text{VLB}}
            $$
            """)
            
            
            st.markdown("展开后可以分解为三项：")
            
            st.markdown(
                """
                $$ \\mathcal{L}_{ \\text{VLB}} = \\underbrace{\\mathbb{E}_q[\\log p_\\theta(x_0|x_1)]}_{L_0} 
            - \\underbrace{\\sum_{t=2}^{T} \\mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t,x_0) \\| p_\\theta(x_{t-1}|x_t))]}_{L_{t-1}} 
            - \\underbrace{D_{KL}(q(x_T|x_0) \\| p(x_T))}_{L_T}
            $$
            """
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**$L_0$：重建项**")
                st.markdown("- 最终步的数据似然")
                st.markdown("- 衡量 $x_0$ 的重建质量")
            
            with col2:
                st.markdown("**$L_{t-1}$：去噪匹配项**")
                st.markdown("- 前向和反向过程的KL散度")
                st.markdown("- 主要的训练目标")
            
            with col3:
                st.markdown("**$L_T$：先验匹配项**")
                st.markdown("- 最终噪声与标准高斯的差距")
                st.markdown("- 通常可以忽略（β足够小时）")
            
            st.markdown("---")
            
            st.markdown("#### 2️⃣ 简化损失函数（DDPM）")
            
            st.markdown("""
            Ho et al. (2020) 发现，简化的损失函数效果更好：
            """)
            
            st.markdown ("""
            $$ \\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t \\sim U(1,T), x_0 \\sim q(x_0), \\epsilon \\sim \\mathcal{N}(0,I)} 
            \\left[ \\| \epsilon - \\epsilon_\\theta(\\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon, t) \\|^2 \\right]
            $$
            """)
            
            st.markdown("**关键简化**：")
            st.markdown("""
            1. **去掉权重系数**：不使用 $\\frac{\\beta_t^2}{2\\sigma_t^2\\alpha_t(1-\\bar{\\alpha}_t)}$ 权重
            2. **直接预测噪声**：而非预测均值
            3. **均匀采样时间步**：$t \\sim U(1,T)$
            """)
            
            st.info("""
            **为什么这样更好？**
            - 简化的损失函数相当于对不同时间步给予相同的重要性
            - 理论上等价于重新加权的ELBO
            - 实践中生成质量更高
            """)
            
            st.markdown("---")
            
            st.markdown("#### 3️⃣ 不同的预测目标")
            
            st.markdown("扩散模型可以预测不同的目标：")
            
            target_type = st.radio(
                "选择预测目标",
                ["预测噪声 ε", "预测原始数据 x₀", "预测得分 ∇log p"]
            )
            
            if "噪声" in target_type:
                st.markdown("**预测噪声 $\\epsilon$** (DDPM原始方法)")
                st.markdown ("$$ \\mathcal{L} = \\mathbb{E} \\left[ \\| \\epsilon - \\epsilon_\\theta(x_t, t) \\|^2 \\right] $$")
                st.markdown("""
                - **优点**：训练稳定，实现简单
                - **缺点**：需要额外计算 $x_0$ 估计
                - **应用**：DDPM, Stable Diffusion
                """)
                
            elif "原始" in target_type:
                st.markdown("**预测原始数据 $x_0$**")
                st.markdown(" $$ \\mathcal{L} = \\mathbb{E} \\left[ \\| x_0 - \\hat{x}_\\theta(x_t, t) \\|^2 \\right] $$")
                st.markdown("""
                - **优点**：直接优化重建质量
                - **缺点**：训练不稳定（特别是小t时）
                - **应用**：某些早期工作
                """)
                
            else:
                st.markdown("**预测得分 $\\nabla_{x_t} \\log p_t(x_t)$** (Score-based模型)")
                st.markdown("$$ \\mathcal{L} = \\mathbb{E} \\left[ \\| \\nabla_{x_t} \\log p_t(x_t) - s_\\theta(x_t, t) \\|^2 \\right] $$")
                st.markdown("""
                - **优点**：统一的理论框架（SDE视角）
                - **缺点**：需要理解score matching
                - **应用**：Score SDE, Imagen
                """)
            
            st.markdown("**三者的关系**：")
            st.markdown("$$ \\epsilon_\\theta(x_t, t) = -\\sqrt{1-\\bar{\\alpha}_t} \\cdot s_\\theta(x_t, t) $$")
            st.markdown("$$ \\hat{x}_\\theta(x_t, t) = \\frac{x_t - \\sqrt{1-\\bar{\\alpha}_t} \\epsilon_\\theta(x_t, t)}{\\sqrt{\\bar{\\alpha}_t}}$$")
        
        with diff_tab2:
            st.markdown("#### 扩散模型的梯度计算")
            
            st.markdown("**训练步骤**：")
            st.code("""
# 1. 采样训练数据
x_0 ~ q(x_0)              # 从数据集采样
t ~ Uniform(1, T)          # 随机采样时间步
ε ~ N(0, I)               # 采样高斯噪声

# 2. 前向扩散（添加噪声）
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

# 3. 模型预测
ε_θ = model(x_t, t)       # 预测噪声

# 4. 计算损失
loss = ||ε - ε_θ||²       # MSE损失

# 5. 反向传播
loss.backward()           # 计算梯度
optimizer.step()          # 更新参数
            """, language="python")
            
            st.markdown("---")
            
            st.markdown("#### 梯度流分析")
            
            st.markdown("**损失函数对模型输出的梯度**：")
            st.markdown("$$ \\frac{\\partial \\mathcal{L}}{\\partial \\epsilon_\\theta} = 2(\\epsilon_\\theta - \\epsilon)$$")
            
            st.markdown("**通过U-Net反向传播**：")
            st.markdown("""
            U-Net是典型的编码器-解码器结构，梯度流经：
            1. **输出层** → 卷积层梯度
            2. **解码器** → 上采样、跳跃连接
            3. **瓶颈层** → 中间表示
            4. **编码器** → 下采样、特征提取
            5. **时间嵌入** → 时间步条件信息
            """)
            
            # 简化示例
            st.markdown("---")
            st.markdown("#### 简化示例：1D扩散模型")
            
            # 参数设置
            T = 100
            t_example = st.slider("选择时间步 t", 1, T, T//2)
            
            # 模拟扩散过程
            beta_start, beta_end = 0.0001, 0.02
            betas = np.linspace(beta_start, beta_end, T)
            alphas = 1 - betas
            alphas_cumprod = np.cumprod(alphas)
            
            # 原始数据（1D）
            x_0 = 2.0
            
            # 生成噪声
            np.random.seed(42)
            epsilon_true = np.random.randn()
            
            # 前向扩散
            sqrt_alpha_t = np.sqrt(alphas_cumprod[t_example-1])
            sqrt_one_minus_alpha_t = np.sqrt(1 - alphas_cumprod[t_example-1])
            x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * epsilon_true
            
            # 模拟模型预测（添加一些误差）
            epsilon_pred = epsilon_true + np.random.randn() * 0.1
            
            # 计算损失
            loss = (epsilon_pred - epsilon_true) ** 2
            
            # 计算梯度
            grad = 2 * (epsilon_pred - epsilon_true)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**前向传播**")
                st.markdown(f"""
                - 原始数据: $x_0 = {x_0:.4f}$
                - 时间步: $t = {t_example}$
                - 真实噪声: $\\epsilon = {epsilon_true:.4f}$
                - 加噪数据: $x_t = {x_t:.4f}$
                - 预测噪声: $\\epsilon_\\theta = {epsilon_pred:.4f}$
                """)
                
                # 可视化
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['x₀', 'x_t', 'ε_true', 'ε_pred'],
                    y=[x_0, x_t, epsilon_true, epsilon_pred],
                    marker_color=['blue', 'orange', 'green', 'red']
                ))
                fig.update_layout(
                    title="数值对比",
                    yaxis_title="值",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**反向传播**")
                st.markdown(f"""
                - 损失: $\\mathcal{{L}} = {loss:.6f}$
                - 梯度: $\\frac{{\\partial \\mathcal{{L}}}}{{\\partial \\epsilon_\\theta}} = {grad:.6f}$
                - 系数: $\\sqrt{{\\bar{{\\alpha}}_t}} = {sqrt_alpha_t:.4f}$
                - 系数: $\\sqrt{{1-\\bar{{\\alpha}}_t}} = {sqrt_one_minus_alpha_t:.4f}$
                """)
                
                # 梯度可视化
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['误差', '梯度'],
                    y=[epsilon_pred - epsilon_true, grad],
                    marker_color=['orange', 'red']
                ))
                fig.update_layout(
                    title="误差与梯度",
                    yaxis_title="值",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**梯度更新**：")
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
            epsilon_updated = epsilon_pred - learning_rate * grad
            
            st.markdown(f"""
            $$\\epsilon_\\theta^{{\\text{{new}}}} = \\epsilon_\\theta - \\eta \\cdot \\frac{{\\partial \\mathcal{{L}}}}{{\\partial \\epsilon_\\theta}} = {epsilon_pred:.4f} - {learning_rate} \\times {grad:.4f} = {epsilon_updated:.4f}$$
            """)
            
            loss_new = (epsilon_updated - epsilon_true) ** 2
            st.metric("损失变化", f"{loss:.6f} → {loss_new:.6f}", 
                     delta=f"{loss_new - loss:.6f}",
                     delta_color="inverse")
        
        with diff_tab3:
            st.markdown("#### 不同训练目标的对比实验")
            
            st.markdown("""
            我们对比三种训练目标在不同时间步的损失权重和梯度行为。
            """)
            
            # 参数
            T = 1000
            betas = np.linspace(0.0001, 0.02, T)
            alphas = 1 - betas
            alphas_cumprod = np.cumprod(alphas)
            
            # 计算不同目标的权重
            t_range = np.arange(1, T+1)
            
            # ELBO权重
            sigma_squared = betas * (1 - np.append(1.0, alphas_cumprod[:-1])) / (1 - alphas_cumprod)
            weight_vlb = betas**2 / (2 * sigma_squared * alphas * (1 - alphas_cumprod))
            
            # Simple权重（均匀）
            weight_simple = np.ones(T)
            
            # SNR权重
            snr = alphas_cumprod / (1 - alphas_cumprod)
            weight_snr = snr
            
            # 可视化
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('损失权重对比', 'SNR vs 时间步', 
                              '累积权重', '梯度尺度')
            )
            
            # 权重对比
            fig.add_trace(go.Scatter(x=t_range, y=weight_vlb / weight_vlb.max(), 
                                    name='VLB权重', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_range, y=weight_simple, 
                                    name='Simple权重', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_range, y=weight_snr / weight_snr.max(), 
                                    name='SNR权重', line=dict(color='green')), row=1, col=1)
            
            # SNR
            fig.add_trace(go.Scatter(x=t_range, y=snr, name='SNR', 
                                    line=dict(color='purple')), row=1, col=2)
            
            # 累积权重
            fig.add_trace(go.Scatter(x=t_range, y=np.cumsum(weight_vlb) / np.sum(weight_vlb), 
                                    name='VLB', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=t_range, y=np.cumsum(weight_simple) / np.sum(weight_simple), 
                                    name='Simple', line=dict(color='red')), row=2, col=1)
            
            # 梯度尺度（近似）
            grad_scale = np.sqrt(1 - alphas_cumprod)
            fig.add_trace(go.Scatter(x=t_range, y=grad_scale, 
                                    name='梯度尺度', line=dict(color='orange')), row=2, col=2)
            
            fig.update_xaxes(title_text="时间步 t")
            fig.update_yaxes(type="log", row=1, col=2)
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**观察**：")
            st.markdown("""
            1. **VLB权重**：后期（大t）权重更高，关注噪声添加阶段
            2. **Simple权重**：所有时间步权重相同，更平衡
            3. **SNR**：信噪比随时间步递减，早期信号强，后期噪声强
            4. **梯度尺度**：后期梯度更大，需要careful调整学习率
            """)
            
            # 实际影响
            st.markdown("---")
            st.markdown("#### 对训练的实际影响")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**VLB加权训练**")
                st.markdown("""
                ✅ 理论上最优（最小化真实ELBO）
                ❌ 后期步骤主导训练
                ❌ 早期重建质量可能较差
                ❌ 训练可能不稳定
                """)
            
            with col2:
                st.markdown("**Simple均匀训练**")
                st.markdown("""
                ✅ 训练稳定
                ✅ 各阶段平衡
                ✅ 实践中效果更好
                ✅ 实现简单
                """)
        
        with diff_tab4:
            st.markdown("#### 数值梯度验证")
            
            st.markdown("""
            通过有限差分法验证解析梯度的正确性，这是调试扩散模型的重要工具。
            """)
            
            # 简化的扩散模型（线性模型，便于理解）
            st.markdown("**简化模型**：线性预测器 $\\epsilon_\\theta(x_t, t) = W \\cdot x_t + b$")
            
            # 参数
            np.random.seed(42)
            W = np.random.randn() * 0.1
            b = np.random.randn() * 0.1
            
            # 数据
            x_0 = 1.5
            t = 50
            T = 100
            
            betas = np.linspace(0.0001, 0.02, T)
            alphas = 1 - betas
            alphas_cumprod = np.cumprod(alphas)
            
            epsilon_true = np.random.randn()
            sqrt_alpha_t = np.sqrt(alphas_cumprod[t-1])
            sqrt_one_minus_alpha_t = np.sqrt(1 - alphas_cumprod[t-1])
            x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * epsilon_true
            
            # 前向传播
            epsilon_pred = W * x_t + b
            loss = (epsilon_pred - epsilon_true) ** 2
            
            # 解析梯度
            grad_epsilon = 2 * (epsilon_pred - epsilon_true)
            grad_W_analytical = grad_epsilon * x_t
            grad_b_analytical = grad_epsilon
            
            # 数值梯度
            epsilon_val = 1e-5
            
            # W的数值梯度
            loss_plus = ((W + epsilon_val) * x_t + b - epsilon_true) ** 2
            loss_minus = ((W - epsilon_val) * x_t + b - epsilon_true) ** 2
            grad_W_numerical = (loss_plus - loss_minus) / (2 * epsilon_val)
            
            # b的数值梯度
            loss_plus = (W * x_t + (b + epsilon_val) - epsilon_true) ** 2
            loss_minus = (W * x_t + (b - epsilon_val) - epsilon_true) ** 2
            grad_b_numerical = (loss_plus - loss_minus) / (2 * epsilon_val)
            
            # 显示结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**W的梯度**")
                st.markdown(f"解析梯度: `{grad_W_analytical:.8f}`")
                st.markdown(f"数值梯度: `{grad_W_numerical:.8f}`")
                diff_W = abs(grad_W_analytical - grad_W_numerical)
                st.markdown(f"差异: `{diff_W:.2e}`")
                
                if diff_W < 1e-7:
                    st.success("✅ 梯度验证通过！")
                elif diff_W < 1e-5:
                    st.warning("⚠️ 有小误差")
                else:
                    st.error("❌ 梯度可能有误")
            
            with col2:
                st.markdown("**b的梯度**")
                st.markdown(f"解析梯度: `{grad_b_analytical:.8f}`")
                st.markdown(f"数值梯度: `{grad_b_numerical:.8f}`")
                diff_b = abs(grad_b_analytical - grad_b_numerical)
                st.markdown(f"差异: `{diff_b:.2e}`")
                
                if diff_b < 1e-7:
                    st.success("✅ 梯度验证通过！")
                elif diff_b < 1e-5:
                    st.warning("⚠️ 有小误差")
                else:
                    st.error("❌ 梯度可能有误")
            
            st.markdown("---")
            
            st.markdown("#### 梯度检查的最佳实践")
            
            st.code("""
def gradient_check(model, x_t, t, epsilon_true, eps=1e-5):
    \"\"\"
    检查扩散模型的梯度正确性
    
    参数:
        model: 扩散模型
        x_t: 加噪输入
        t: 时间步
        epsilon_true: 真实噪声
        eps: 有限差分步长
    \"\"\"
    # 前向传播
    epsilon_pred = model(x_t, t)
    loss = ((epsilon_pred - epsilon_true) ** 2).mean()
    
    # 解析梯度
    loss.backward()
    analytical_grads = [p.grad.clone() for p in model.parameters()]
    
    # 数值梯度
    numerical_grads = []
    for param in model.parameters():
        param_grad = torch.zeros_like(param)
        
        # 遍历每个参数
        it = np.nditer(param.data.cpu().numpy(), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            
            # f(x + eps)
            old_val = param.data[idx].item()
            param.data[idx] = old_val + eps
            loss_plus = ((model(x_t, t) - epsilon_true) ** 2).mean()
            
            # f(x - eps)
            param.data[idx] = old_val - eps
            loss_minus = ((model(x_t, t) - epsilon_true) ** 2).mean()
            
            # 数值梯度
            param_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            # 恢复原值
            param.data[idx] = old_val
            it.iternext()
        
        numerical_grads.append(param_grad)
    
    # 比较
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = torch.abs(a_grad - n_grad).max().item()
        print(f"参数 {i}: 最大差异 = {diff:.2e}")
        
    return analytical_grads, numerical_grads
            """, language="python")
            
            st.info("""
            **注意事项**：
            - 使用双精度浮点数（float64）进行检查
            - eps通常选择1e-5到1e-7之间
            - 只在小模型和小批量上进行检查（计算开销大）
            - 检查通过后，可以用单精度训练
            """)

# TAB 7: 交互实验室
with tab7:
    st.header("🎮 神经网络交互实验室")

    experiment_type = st.selectbox(
        "选择实验类型",
        [
            "CNN特征图可视化",
            "GNN节点分类演示",
            "激活函数对比",
            "优化器轨迹可视化",
            "损失函数3D地形图",
            "🚀 批量参数对比",
        ],
    )

    if experiment_type == "CNN特征图可视化":
        st.markdown("### CNN卷积特征图实时可视化")

        # 图像上传选项
        st.markdown("#### 📁 图像输入方式")
        input_method = st.radio("选择输入方式", ["上传真实图像", "使用示例图像"])

        input_image = None
        original_size = None

        if input_method == "上传真实图像":
            uploaded_file = st.file_uploader(
                "上传图像 (支持 JPG, PNG, GIF)",
                type=["jpg", "jpeg", "png", "gif"],
                help="上传你自己的图像来查看CNN处理效果",
            )

            if uploaded_file is not None:
                # 读取并处理上传的图像
                from PIL import Image
                import io

                try:
                    image = Image.open(uploaded_file)
                    original_size = image.size
                    st.markdown(
                        f"**原始图像尺寸**: {original_size[0]} × {original_size[1]}"
                    )

                    # 转换为灰度图像
                    if image.mode != "L":
                        image = image.convert("L")

                    # 调整大小以便处理
                    target_size = st.slider("处理尺寸", 64, 256, 128)
                    image = image.resize(
                        (target_size, target_size), Image.Resampling.LANCZOS
                    )

                    # 转换为numpy数组
                    input_image = np.array(image, dtype=np.float32) / 255.0

                    # 显示原始图像
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("#### 📷 原始图像")
                        st.image(
                            uploaded_file,
                            caption="上传的原始图像",
                            use_container_width=True,
                        )

                    with col2:
                        st.markdown("#### 🔧 处理后图像")
                        fig = px.imshow(
                            input_image,
                            color_continuous_scale="gray",
                            title=f"处理后 ({target_size}×{target_size})",
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.error(f"图像处理失败: {str(e)}")
                    st.info("请尝试上传其他格式的图像")
                    input_method = "使用示例图像"

        if input_method == "使用示例图像" or input_image is None:
            # 创建示例图像
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown("#### 🎨 示例图像选择")
                pattern_type = st.selectbox(
                    "选择图案类型", ["正弦波纹", "同心圆", "随机噪声", "棋盘格", "渐变"]
                )

                image_size = st.slider("图像尺寸", 64, 256, 128)

                # 生成不同类型的示例图像
                if pattern_type == "正弦波纹":
                    x = np.linspace(-5, 5, image_size)
                    y = np.linspace(-5, 5, image_size)
                    X, Y = np.meshgrid(x, y)
                    input_image = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(
                        image_size, image_size
                    )

                elif pattern_type == "同心圆":
                    x = np.linspace(-1, 1, image_size)
                    y = np.linspace(-1, 1, image_size)
                    X, Y = np.meshgrid(x, y)
                    R = np.sqrt(X**2 + Y**2)
                    input_image = np.sin(10 * R) + 0.1 * np.random.randn(
                        image_size, image_size
                    )

                elif pattern_type == "随机噪声":
                    input_image = np.random.randn(image_size, image_size) * 0.5

                elif pattern_type == "棋盘格":
                    input_image = np.zeros((image_size, image_size))
                    block_size = image_size // 8
                    for i in range(0, image_size, block_size):
                        for j in range(0, image_size, block_size):
                            if (i // block_size + j // block_size) % 2 == 0:
                                input_image[i : i + block_size, j : j + block_size] = (
                                    1.0
                                )

                elif pattern_type == "渐变":
                    x = np.linspace(0, 1, image_size)
                    y = np.linspace(0, 1, image_size)
                    X, Y = np.meshgrid(x, y)
                    input_image = (
                        X + Y * 0.5 + 0.1 * np.random.randn(image_size, image_size)
                    )

                # 归一化到[0,1]
                input_image = (input_image - input_image.min()) / (
                    input_image.max() - input_image.min()
                )

                fig = px.imshow(
                    input_image, color_continuous_scale="gray", title="示例图像"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")

        # 只有在有图像时才显示卷积处理
        if input_image is not None:
            # 卷积核配置
            st.markdown("#### 🔧 卷积核配置")
            col2, col3 = st.columns([1, 1])

            with col2:
                kernel_type = st.radio(
                    "卷积核类型", ["边缘检测", "模糊", "锐化", "自定义"]
                )

                if kernel_type == "边缘检测":
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                elif kernel_type == "模糊":
                    kernel_size = st.slider("模糊核大小", 3, 11, 5)
                    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
                elif kernel_type == "锐化":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                else:  # 自定义
                    st.markdown("**自定义卷积核**")
                    kernel_size = st.slider("核大小", 3, 5, 3)
                    kernel_values = []
                    for i in range(kernel_size):
                        row = []
                        for j in range(kernel_size):
                            val = st.number_input(
                                f"K[{i},{j}]", value=0.0, key=f"kernel_{i}_{j}"
                            )
                            row.append(val)
                        kernel_values.append(row)
                    kernel = np.array(kernel_values)

                # 显示卷积核
                fig = px.imshow(kernel, color_continuous_scale="RdBu", title="卷积核")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")

                # 卷积参数
                st.markdown("**卷积参数**")
                padding = st.slider("填充", 0, 2, 0)
                stride = st.slider("步长", 1, 3, 1)

            with col3:
                st.markdown("#### 🎯 特征图结果")

                # 应用卷积
                if kernel_type == "模糊" and kernel_size > 3:
                    # 大核使用valid模式避免边界问题
                    feature_map = signal.convolve2d(input_image, kernel, mode="valid")
                else:
                    feature_map = signal.convolve2d(input_image, kernel, mode="same")

                # 应用激活函数
                activation = st.selectbox("激活函数", ["无", "ReLU", "Sigmoid", "Tanh"])
                if activation == "ReLU":
                    feature_map = np.maximum(0, feature_map)
                elif activation == "Sigmoid":
                    feature_map = 1 / (1 + np.exp(-feature_map))
                elif activation == "Tanh":
                    feature_map = np.tanh(feature_map)

                # 显示特征图
                fig = px.imshow(
                    feature_map, color_continuous_scale="viridis", title="输出特征图"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")

                # 特征图统计信息
                st.markdown("**特征图统计**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("最小值", f"{feature_map.min():.3f}")
                    st.metric("最大值", f"{feature_map.max():.3f}")
                with col_b:
                    st.metric("均值", f"{feature_map.mean():.3f}")
                    st.metric("标准差", f"{feature_map.std():.3f}")

            # 多层特征图演化
            st.markdown("### 🔄 多层卷积演化")
            num_layers = st.slider("卷积层数", 1, 5, 3)

            layers = []
            current = input_image.copy()

            for i in range(num_layers):
                # 每层使用不同的卷积核
                if i % 3 == 0:
                    layer_kernel = np.array(
                        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
                    )  # 边缘检测
                elif i % 3 == 1:
                    layer_kernel = np.ones((3, 3)) / 9  # 模糊
                else:
                    layer_kernel = np.array(
                        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
                    )  # 锐化

                # 应用卷积和激活
                current = signal.convolve2d(current, layer_kernel, mode="same")
                current = np.maximum(0, current)  # ReLU
                layers.append(current.copy())

            # 可视化所有层
            fig = go.Figure()

            for i, layer in enumerate(layers):
                fig.add_trace(
                    go.Heatmap(
                        z=layer,
                        colorscale="viridis",
                        name=f"Layer {i+1}",
                        showscale=False if i < num_layers - 1 else True,
                    )
                )

            fig.update_layout(
                title=f"CNN多层特征演化 ({num_layers}层)",
                height=400,
                xaxis_title="Width",
                yaxis_title="Height",
            )

            st.plotly_chart(fig, width="stretch")

            # 输出尺寸计算
            st.markdown("### 📏 输出尺寸计算")
            input_h, input_w = input_image.shape
            kernel_h, kernel_w = kernel.shape

            output_h = (input_h + 2 * padding - kernel_h) // stride + 1
            output_w = (input_w + 2 * padding - kernel_w) // stride + 1

            st.markdown(
                f"""
            **计算过程**:
            - 输入尺寸: {input_h} × {input_w}
            - 卷积核: {kernel_h} × {kernel_w}
            - 填充: {padding}
            - 步长: {stride}
            - 输出尺寸: {output_h} × {output_w}
            
            **公式**: 
            $H_{{out}} = \left\lfloor \frac{{H_{{in}} + 2P - K}}{{S}} \right\rfloor + 1$
            """
            )

            # 实时对比不同卷积核效果
            st.markdown("### ⚡ 实时卷积核对比")

            # 预定义的卷积核集合
            kernels_dict = {
                "边缘检测": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                "高斯模糊": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
                "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
                "浮雕": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
                "轮廓": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            }

            selected_kernels = st.multiselect(
                "选择要对比的卷积核",
                list(kernels_dict.keys()),
                default=["边缘检测", "高斯模糊", "锐化"],
            )

            if selected_kernels:
                # 创建对比图表
                cols = len(selected_kernels)
                fig = make_subplots(
                    rows=1,
                    cols=cols,
                    subplot_titles=selected_kernels,
                    specs=[[{"type": "heatmap"} for _ in range(cols)]],
                )

                for i, kernel_name in enumerate(selected_kernels):
                    kernel = kernels_dict[kernel_name]
                    result = signal.convolve2d(input_image, kernel, mode="same")
                    result = np.maximum(0, result)  # ReLU激活

                    fig.add_trace(
                        go.Heatmap(z=result, colorscale="viridis", showscale=False),
                        row=1,
                        col=i + 1,
                    )

                fig.update_layout(
                    title="不同卷积核效果对比", height=300, showlegend=False
                )

                st.plotly_chart(fig, width="stretch")
        else:
            st.info("👆 请先上传图像或选择示例图像来开始CNN处理")

        # 多层特征图演化
        st.markdown("### 多层特征图演化")
        num_layers = st.slider("网络层数", 1, 5, 3)

        layers = []
        current = input_image

        for i in range(num_layers):
            # 随机卷积核
            kernel = np.random.randn(3, 3) * 0.3
            current = signal.convolve2d(current, kernel, mode="same")
            current = np.maximum(0, current)  # ReLU
            layers.append(current)

        # 可视化所有层
        fig = go.Figure()

        for i, layer in enumerate(layers):
            fig.add_trace(
                go.Heatmap(
                    z=layer,
                    colorscale="viridis",
                    name=f"Layer {i+1}",
                    showscale=False if i < num_layers - 1 else True,
                )
            )

        # 创建子图布局
        fig.update_layout(
            title=(
                "CNN多层特征图演化"
                if CHINESE_SUPPORTED
                else "CNN Multi-layer Feature Maps Evolution"
            ),
            height=400,
            xaxis_title="Width",
            yaxis_title="Height",
        )

        st.plotly_chart(fig, width="stretch")

    elif experiment_type == "GNN节点分类演示":
        st.markdown("### GNN节点分类实时演示")

        # 创建图数据
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 图结构设置")
            num_nodes = st.slider("节点数量", 4, 12, 6)
            edge_prob = st.slider("边连接概率", 0.1, 0.8, 0.3)
            num_classes = st.slider("类别数量", 2, 4, 3)

            # 生成随机图
            G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=42)
            pos = nx.spring_layout(G, seed=42)

            # 随机分配节点特征和标签
            node_features = np.random.randn(num_nodes, 4)
            node_labels = np.random.randint(0, num_classes, num_nodes)

            # 可视化图结构
            colors = ["red", "blue", "green", "orange"]
            node_colors = [colors[label] for label in node_labels]

            # 交互式图可视化
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node {node}<br>Label: {node_labels[node]}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[str(i) for i in range(len(node_x))],
            hovertext=node_text,
            textposition="middle center",
            marker=dict(size=20, color=node_colors, line_width=2),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=(
                    "图结构与真实标签"
                    if CHINESE_SUPPORTED
                    else "Graph Structure and True Labels"
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
            ),
        )
        st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("#### GNN训练过程")

            # 简化的GNN训练
            num_epochs = st.slider("训练轮数", 10, 200, 50)
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01)

            # 初始化权重
            W = np.random.randn(4, num_classes) * 0.1

            # 训练过程记录
            losses = []
            accuracies = []

            # 邻接矩阵
            A = nx.adjacency_matrix(G).todense()
            A = A + np.eye(num_nodes)  # 自环

            for epoch in range(num_epochs):
                # 前向传播
                H = A @ node_features @ W
                predictions = np.argmax(H, axis=1)

                # 计算损失
                one_hot = np.eye(num_classes)[node_labels]
                loss = np.mean((H - one_hot) ** 2)
                losses.append(loss)

                # 计算准确率
                accuracy = np.mean(predictions == node_labels)
                accuracies.append(accuracy)

                # 反向传播（简化版）
                grad = 2 * (H - one_hot) / num_nodes
                W -= learning_rate * node_features.T @ A @ grad

            # 可视化训练过程
            fig = go.Figure()

            # 损失曲线
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(losses))),
                    y=losses,
                    mode="lines",
                    name="训练损失" if CHINESE_SUPPORTED else "Training Loss",
                    line=dict(color="blue"),
                )
            )

            # 准确率曲线
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracies))),
                    y=accuracies,
                    mode="lines",
                    name=(
                        "分类准确率" if CHINESE_SUPPORTED else "Classification Accuracy"
                    ),
                    line=dict(color="red"),
                    yaxis="y2",
                )
            )

            fig.update_layout(
                title="GNN训练过程" if CHINESE_SUPPORTED else "GNN Training Process",
                xaxis_title="Epoch",
                yaxis=dict(title="Loss", side="left"),
                yaxis2=dict(title="Accuracy", side="right", overlaying="y"),
                height=400,
                legend=dict(x=0.01, y=0.99),
            )

            st.plotly_chart(fig, width="stretch")

            # 最终预测可视化
            final_predictions = np.argmax(A @ node_features @ W, axis=1)
            pred_colors = [colors[pred] for pred in final_predictions]

            # GNN预测结果可视化
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []

        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(
                f"Node {node}<br>True: {node_labels[node]}<br>Pred: {final_predictions[i]}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[str(i) for i in range(len(node_x))],
            hovertext=node_text,
            textposition="middle center",
            marker=dict(size=20, color=pred_colors, line_width=2),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="GNN预测结果" if CHINESE_SUPPORTED else "GNN Prediction Results",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
            ),
        )
        st.plotly_chart(fig, width="stretch")

        st.metric(
            "最终准确率" if CHINESE_SUPPORTED else "Final Accuracy",
            f"{accuracies[-1]:.3f}",
        )

    elif experiment_type == "激活函数对比":
        st.markdown("### 激活函数交互式对比")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 激活函数选择")
            activations = st.multiselect(
                "选择激活函数",
                ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "ELU", "Swish"],
                default=["ReLU", "Sigmoid", "Tanh"],
            )

            x_range = st.slider("x范围", -10, 10, 5)
            num_points = st.slider("采样点数", 50, 500, 200)

            # 生成x值
            x = np.linspace(-x_range, x_range, num_points)

            # 定义激活函数
            def relu(x):
                return np.maximum(0, x)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            def tanh(x):
                return np.tanh(x)

            def leaky_relu(x):
                return np.where(x > 0, x, 0.01 * x)

            def elu(x):
                return np.where(x > 0, x, np.exp(x) - 1)

            def swish(x):
                return x * sigmoid(x)

            # 计算激活函数值
            activation_funcs = {
                "ReLU": relu,
                "Sigmoid": sigmoid,
                "Tanh": tanh,
                "Leaky ReLU": leaky_relu,
                "ELU": elu,
                "Swish": swish,
            }

        with col2:
            st.markdown("#### 激活函数图像")

            fig = go.Figure()

            for act_name in activations:
                y = activation_funcs[act_name](x)
                fig.add_trace(go.Scatter(x=x, y=y, name=act_name, mode="lines"))

            fig.update_layout(
                title=(
                    "激活函数对比"
                    if CHINESE_SUPPORTED
                    else "Activation Functions Comparison"
                ),
                xaxis_title="x",
                yaxis_title="f(x)",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

        # 导数对比
        st.markdown("#### 激活函数导数对比")

        # 计算导数
        def derivative(f, x, h=1e-5):
            return (f(x + h) - f(x - h)) / (2 * h)

        fig = go.Figure()

        for act_name in activations:
            f = activation_funcs[act_name]
            dy = derivative(f, x)
            fig.add_trace(go.Scatter(x=x, y=dy, name=f"{act_name} 导数", mode="lines"))

        fig.update_layout(
            title=(
                "激活函数导数对比"
                if CHINESE_SUPPORTED
                else "Activation Functions Derivatives Comparison"
            ),
            xaxis_title="x",
            yaxis_title="f'(x)",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        # 梯度消失/爆炸分析
        st.markdown("#### 梯度传播分析")

        depth = st.slider("网络深度", 5, 50, 20)
        input_val = st.slider("输入值", -2.0, 2.0, 1.0)

        gradients = {}
        for act_name in activations:
            f = activation_funcs[act_name]
            grad = input_val
            grad_history = [grad]

            for _ in range(depth):
                grad = grad * derivative(f, grad)
                grad_history.append(grad)

            gradients[act_name] = grad_history

        fig = go.Figure()
        for act_name in activations:
            fig.add_trace(
                go.Scatter(
                    x=list(range(depth + 1)),
                    y=np.log10(np.abs(gradients[act_name]) + 1e-10),
                    name=act_name,
                    mode="lines+markers",
                )
            )

        fig.update_layout(
            title=(
                f"梯度传播 (输入={input_val}, 深度={depth})"
                if CHINESE_SUPPORTED
                else f"Gradient Propagation (input={input_val}, depth={depth})"
            ),
            xaxis_title="层数" if CHINESE_SUPPORTED else "Layer",
            yaxis_title="log|梯度|",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    elif experiment_type == "优化器轨迹可视化":
        st.markdown("### 优化器轨迹3D可视化")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 损失函数设置")
            loss_function = st.selectbox(
                "选择损失函数",
                ["二次函数", "Rosenbrock函数", "Himmelblau函数", "Ackley函数"],
            )

            # 定义损失函数
            def quadratic(x, y):
                return x**2 + y**2

            def rosenbrock(x, y):
                return (1 - x) ** 2 + 100 * (y - x**2) ** 2

            def himmelblau(x, y):
                return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

            def ackley(x, y):
                return (
                    -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
                    - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
                    + np.e
                    + 20
                )

            loss_funcs = {
                "二次函数": quadratic,
                "Rosenbrock函数": rosenbrock,
                "Himmelblau函数": himmelblau,
                "Ackley函数": ackley,
            }

            func = loss_funcs[loss_function]

            # 优化器设置
            optimizer = st.selectbox("优化器", ["SGD", "Momentum", "Adam", "RMSprop"])
            learning_rate = st.slider("学习率", 0.001, 0.1, 0.01)
            iterations = st.slider("迭代次数", 50, 500, 200)

            # 起始点
            start_x = st.slider("起始x", -5.0, 5.0, 3.0)
            start_y = st.slider("起始y", -5.0, 5.0, 3.0)

        with col2:
            st.markdown("#### 优化轨迹")

            # 优化器实现
            def optimize(func, start_x, start_y, lr, iterations, optimizer):
                x, y = start_x, start_y
                trajectory = [(x, y, func(x, y))]

                if optimizer == "SGD":
                    for i in range(iterations):
                        # 数值梯度
                        h = 1e-5
                        grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
                        grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)

                        x -= lr * grad_x
                        y -= lr * grad_y
                        trajectory.append((x, y, func(x, y)))

                elif optimizer == "Momentum":
                    vx, vy = 0, 0
                    momentum = 0.9
                    for i in range(iterations):
                        h = 1e-5
                        grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
                        grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)

                        vx = momentum * vx - lr * grad_x
                        vy = momentum * vy - lr * grad_y

                        x += vx
                        y += vy
                        trajectory.append((x, y, func(x, y)))

                elif optimizer == "Adam":
                    m_x, m_y = 0, 0
                    v_x, v_y = 0, 0
                    beta1, beta2 = 0.9, 0.999
                    epsilon = 1e-8
                    t = 0

                    for i in range(iterations):
                        t += 1
                        h = 1e-5
                        grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
                        grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)

                        m_x = beta1 * m_x + (1 - beta1) * grad_x
                        m_y = beta1 * m_y + (1 - beta1) * grad_y

                        v_x = beta2 * v_x + (1 - beta2) * grad_x**2
                        v_y = beta2 * v_y + (1 - beta2) * grad_y**2

                        m_x_hat = m_x / (1 - beta1**t)
                        m_y_hat = m_y / (1 - beta1**t)
                        v_x_hat = v_x / (1 - beta2**t)
                        v_y_hat = v_y / (1 - beta2**t)

                        x -= lr * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
                        y -= lr * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
                        trajectory.append((x, y, func(x, y)))

                else:  # RMSprop
                    v_x, v_y = 0, 0
                    beta = 0.9
                    epsilon = 1e-8
                    for i in range(iterations):
                        h = 1e-5
                        grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
                        grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)

                        v_x = beta * v_x + (1 - beta) * grad_x**2
                        v_y = beta * v_y + (1 - beta) * grad_y**2

                        x -= lr * grad_x / (np.sqrt(v_x) + epsilon)
                        y -= lr * grad_y / (np.sqrt(v_y) + epsilon)
                        trajectory.append((x, y, func(x, y)))

                return trajectory

            # 运行优化
            trajectory = optimize(
                func, start_x, start_y, learning_rate, iterations, optimizer
            )

            # 3D可视化
            x_range = np.linspace(-5, 5, 50)
            y_range = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = func(X, Y)

            fig = go.Figure(
                data=[
                    go.Surface(x=X, y=Y, z=Z, opacity=0.8, colorscale="Viridis"),
                    go.Scatter3d(
                        x=[point[0] for point in trajectory],
                        y=[point[1] for point in trajectory],
                        z=[point[2] for point in trajectory],
                        mode="markers+lines",
                        marker=dict(size=5, color="red"),
                        line=dict(color="red", width=3),
                        name="优化轨迹",
                    ),
                ]
            )

            fig.update_layout(
                title=(
                    f"{optimizer} 优化轨迹"
                    if CHINESE_SUPPORTED
                    else f"{optimizer} Optimization Trajectory"
                ),
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="Loss"),
                height=500,
            )
            st.plotly_chart(fig, width="stretch")

            # 损失变化
            losses = [point[2] for point in trajectory]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=list(range(len(losses))), y=losses, mode="lines")
            )
            fig.update_layout(
                title="损失变化" if CHINESE_SUPPORTED else "Loss Change",
                xaxis_title="迭代次数" if CHINESE_SUPPORTED else "Iteration",
                yaxis_title="Loss",
                height=300,
            )
            st.plotly_chart(fig, width="stretch")

            st.metric(
                "最终损失" if CHINESE_SUPPORTED else "Final Loss", f"{losses[-1]:.4f}"
            )
            st.metric(
                "最终位置" if CHINESE_SUPPORTED else "Final Position",
                f"({trajectory[-1][0]:.3f}, {trajectory[-1][1]:.3f})",
            )

    elif experiment_type == "损失函数3D地形图":
        st.markdown("### 损失函数3D地形图交互探索")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 参数设置")

            # 选择损失函数
            loss_type = st.selectbox(
                "损失函数",
                ["二次函数", "Rosenbrock函数", "Beale函数", "Booth函数", "Matyas函数"],
            )

            # 定义损失函数
            def beale(x, y):
                return (
                    (1.5 - x + x * y) ** 2
                    + (2.25 - x + x * y**2) ** 2
                    + (2.625 - x + x * y**3) ** 2
                )

            def booth(x, y):
                return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

            def matyas(x, y):
                return 0.26 * (x**2 + y**2) - 0.48 * x * y

            loss_functions = {
                "二次函数": lambda x, y: x**2 + y**2,
                "Rosenbrock函数": lambda x, y: (1 - x) ** 2 + 100 * (y - x**2) ** 2,
                "Beale函数": beale,
                "Booth函数": booth,
                "Matyas函数": matyas,
            }

            func = loss_functions[loss_type]

            # 视角控制
            elevation = st.slider("仰角", 0, 90, 30)
            azimuth = st.slider("方位角", 0, 360, 45)

            # 范围控制
            x_range = st.slider("x范围", 2, 10, 5)
            y_range = st.slider("y范围", 2, 10, 5)
            resolution = st.slider("分辨率", 20, 100, 50)

        with col2:
            st.markdown("#### 3D地形图")

            # 生成网格
            x = np.linspace(-x_range, x_range, resolution)
            y = np.linspace(-y_range, y_range, resolution)
            X, Y = np.meshgrid(x, y)
            Z = func(X, Y)

            # 创建3D图
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z,
                        colorscale="Viridis",
                        colorbar=dict(title="Loss Value"),
                        contours=dict(
                            z=dict(
                                show=True,
                                usecolormap=True,
                                highlightcolor="limegreen",
                                project_z=True,
                            )
                        ),
                    )
                ]
            )

            fig.update_layout(
                title=(
                    f"{loss_type} 3D地形图"
                    if CHINESE_SUPPORTED
                    else f"{loss_type} 3D Landscape"
                ),
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="Loss",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
                ),
                height=600,
            )

            # 更新视角
            fig.update_layout(
                scene_camera=dict(
                    eye=dict(
                        x=np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)),
                        y=np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)),
                        z=np.sin(np.radians(elevation)),
                    )
                )
            )

            st.plotly_chart(fig, width="stretch")

        # 等高线图
        st.markdown("#### 等高线图")

        # 等高线图
        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
                colorscale="Viridis",
            )
        )

        fig.update_layout(
            title="等高线图" if CHINESE_SUPPORTED else "Contour Plot",
            xaxis_title="x",
            yaxis_title="y",
            height=400,
        )

        st.plotly_chart(fig, width="stretch")

        # 填充等高线图
        fig2 = go.Figure()

        fig2.add_trace(go.Contour(x=x, y=y, z=Z, colorscale="Viridis", showscale=True))

        fig2.update_layout(
            title="填充等高线图" if CHINESE_SUPPORTED else "Filled Contour Plot",
            xaxis_title="x",
            yaxis_title="y",
            height=400,
        )

        st.plotly_chart(fig2, width="stretch")

        # 梯度场可视化
        st.markdown("#### 梯度场可视化")

        # 计算梯度
        grad_x, grad_y = np.gradient(Z, x, y)

        # 降采样以便清晰显示
        skip = max(1, resolution // 20)
        x_sub = X[::skip, ::skip]
        y_sub = Y[::skip, ::skip]
        grad_x_sub = grad_x[::skip, ::skip]
        grad_y_sub = grad_y[::skip, ::skip]

        # 梯度场可视化
        fig = go.Figure()

        # 等高线背景
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=Z,
                contours=dict(showlabels=False, start=0, end=Z.max(), size=15),
                colorscale="Viridis",
                opacity=0.3,
                showscale=False,
                name="等高线",
            )
        )

        # 梯度向量场
        magnitude = np.sqrt(grad_x_sub**2 + grad_y_sub**2)

        fig.add_trace(
            go.Scatter(
                x=x_sub.flatten(),
                y=y_sub.flatten(),
                mode="markers",
                marker=dict(
                    size=8,
                    color=magnitude.flatten(),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="梯度大小"),
                ),
                name="梯度点",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>梯度: %{marker.color:.3f}<extra></extra>",
            )
        )

        # 添加梯度向量箭头（使用注释）
        for i in range(len(x_sub.flatten())):
            xi = x_sub.flatten()[i]
            yi = y_sub.flatten()[i]
            dxi = -grad_x_sub.flatten()[i] * 0.1  # 缩放因子
            dyi = -grad_y_sub.flatten()[i] * 0.1

            fig.add_annotation(
                x=xi,
                y=yi,
                ax=xi + dxi,
                ay=yi + dyi,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
            )

        fig.update_layout(
            title=(
                "梯度场（负梯度方向）"
                if CHINESE_SUPPORTED
                else "Gradient Field (Negative Gradient Direction)"
            ),
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            showlegend=True,
        )

        st.plotly_chart(fig, width="stretch")

    elif experiment_type == "🚀 批量参数对比":
        st.markdown("### 🚀 高效批量参数对比工具")
        st.markdown("**快速对比不同参数组合的效果，找到最优配置**")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📊 对比类型选择")
            comparison_type = st.selectbox(
                "选择对比类型",
                [
                    "优化器性能对比",
                    "学习率影响分析",
                    "网络深度对比",
                    "激活函数性能",
                    "正则化效果对比",
                ],
            )

            st.markdown("#### ⚙️ 参数配置")
            num_configs = st.slider("对比配置数量", 2, 8, 4)

            # 根据对比类型显示不同的参数配置
            if comparison_type == "优化器性能对比":
                st.markdown("**优化器配置**")
                optimizers = st.multiselect(
                    "选择优化器进行对比",
                    ["SGD", "Momentum", "Adam", "RMSprop", "AdaGrad", "Nesterov"],
                    default=["SGD", "Adam", "RMSprop"],
                )

                learning_rates = st.multiselect(
                    "学习率", [0.001, 0.01, 0.1, 0.0001], default=[0.001, 0.01]
                )

                epochs = st.slider("训练轮数", 50, 200, 100)

            elif comparison_type == "学习率影响分析":
                st.markdown("**学习率配置**")
                lr_min = st.number_input("最小学习率", value=0.0001, format="%.4f")
                lr_max = st.number_input("最大学习率", value=0.1, format="%.4f")
                num_lr = st.slider("学习率数量", 3, 10, 5)

                # 生成对数空间的学习率
                learning_rates = np.logspace(np.log10(lr_min), np.log10(lr_max), num_lr)
                optimizer = st.selectbox("优化器", ["Adam", "SGD", "RMSprop"])
                epochs = st.slider("训练轮数", 50, 200, 100)

            elif comparison_type == "网络深度对比":
                st.markdown("**网络配置**")
                depths = st.multiselect(
                    "网络层数", [2, 3, 4, 5, 6, 8, 10], default=[2, 4, 6, 8]
                )

                hidden_dims = st.selectbox("隐藏层维度", [32, 64, 128, 256])
                learning_rate = st.slider("学习率", 0.001, 0.1, 0.01)
                epochs = st.slider("训练轮数", 50, 200, 100)

            elif comparison_type == "激活函数性能":
                st.markdown("**激活函数配置**")
                activations = st.multiselect(
                    "激活函数",
                    ["ReLU", "Leaky ReLU", "ELU", "Swish", "GELU", "Tanh", "Sigmoid"],
                    default=["ReLU", "Leaky ReLU", "ELU", "Swish"],
                )

                learning_rate = st.slider("学习率", 0.001, 0.1, 0.01)
                epochs = st.slider("训练轮数", 50, 200, 100)

            elif comparison_type == "正则化效果对比":
                st.markdown("**正则化配置**")
                dropout_rates = st.multiselect(
                    "Dropout率", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=[0.0, 0.2, 0.4]
                )

                l2_regs = st.multiselect(
                    "L2正则化系数",
                    [0.0, 0.001, 0.01, 0.1, 1.0],
                    default=[0.0, 0.01, 0.1],
                )

                learning_rate = st.slider("学习率", 0.001, 0.1, 0.01)
                epochs = st.slider("训练轮数", 50, 200, 100)

        with col2:
            st.markdown("#### 🎯 批量运行")

            if st.button("🚀 开始批量对比", type="primary"):
                # 显示进度条
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = {}

                if comparison_type == "优化器性能对比":
                    for i, optimizer in enumerate(optimizers):
                        for j, lr in enumerate(learning_rates):
                            progress = (i * len(learning_rates) + j + 1) / (
                                len(optimizers) * len(learning_rates)
                            )
                            progress_bar.progress(progress)
                            status_text.text(
                                f"正在测试 {optimizer} (lr={lr})... {progress:.1%}"
                            )

                            # 模拟训练过程
                            result = simulate_training(
                                optimizer=optimizer,
                                learning_rate=lr,
                                epochs=epochs,
                                comparison_type=comparison_type,
                            )
                            all_results[f"{optimizer}_lr{lr}"] = result

                elif comparison_type == "学习率影响分析":
                    for i, lr in enumerate(learning_rates):
                        progress = (i + 1) / len(learning_rates)
                        progress_bar.progress(progress)
                        status_text.text(f"正在测试学习率 {lr:.4f}... {progress:.1%}")

                        result = simulate_training(
                            optimizer=optimizer,
                            learning_rate=lr,
                            epochs=epochs,
                            comparison_type=comparison_type,
                        )
                        all_results[f"lr{lr:.4f}"] = result

                elif comparison_type == "网络深度对比":
                    for i, depth in enumerate(depths):
                        progress = (i + 1) / len(depths)
                        progress_bar.progress(progress)
                        status_text.text(f"正在测试 {depth}层网络... {progress:.1%}")

                        result = simulate_training(
                            depth=depth,
                            hidden_dim=hidden_dims,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            comparison_type=comparison_type,
                        )
                        all_results[f"{depth}layers"] = result

                elif comparison_type == "激活函数性能":
                    for i, activation in enumerate(activations):
                        progress = (i + 1) / len(activations)
                        progress_bar.progress(progress)
                        status_text.text(f"正在测试 {activation}... {progress:.1%}")

                        result = simulate_training(
                            activation=activation,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            comparison_type=comparison_type,
                        )
                        all_results[activation] = result

                elif comparison_type == "正则化效果对比":
                    for i, dropout in enumerate(dropout_rates):
                        for j, l2 in enumerate(l2_regs):
                            progress = (i * len(l2_regs) + j + 1) / (
                                len(dropout_rates) * len(l2_regs)
                            )
                            progress_bar.progress(progress)
                            status_text.text(
                                f"正在测试 Dropout={dropout}, L2={l2}... {progress:.1%}"
                            )

                            result = simulate_training(
                                dropout=dropout,
                                l2_reg=l2,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                comparison_type=comparison_type,
                            )
                            all_results[f"dropout{dropout}_l2{l2}"] = result

                progress_bar.progress(1.0)
                status_text.text("✅ 批量测试完成！")

                # 显示结果
                st.session_state.comparison_results = all_results
                st.session_state.comparison_type = comparison_type

        # 显示对比结果
        if "comparison_results" in st.session_state:
            st.markdown("---")
            st.markdown("### 📈 对比结果分析")

            results = st.session_state.comparison_results
            comp_type = st.session_state.comparison_type

            # 创建对比图表
            fig = go.Figure()

            # 损失曲线对比
            for name, result in results.items():
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(result["losses"]))),
                        y=result["losses"],
                        mode="lines",
                        name=f"{name} (最终损失: {result['final_loss']:.4f})",
                        line=dict(width=2),
                    )
                )

            fig.update_layout(
                title=f"{comp_type} - 损失曲线对比",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=500,
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

            # 性能排名表
            st.markdown("#### 🏆 性能排名")

            # 计算排名
            sorted_results = sorted(results.items(), key=lambda x: x[1]["final_loss"])

            ranking_data = []
            for i, (name, result) in enumerate(sorted_results):
                ranking_data.append(
                    {
                        "排名": i + 1,
                        "配置": name,
                        "最终损失": f"{result['final_loss']:.4f}",
                        "收敛速度": f"{result['convergence_epoch']} epochs",
                        "最终准确率": f"{result['final_accuracy']:.3f}",
                        "训练时间": f"{result['training_time']:.2f}s",
                    }
                )

            df_ranking = pd.DataFrame(ranking_data)
            st.dataframe(df_ranking, use_container_width=True)

            # 最佳配置推荐
            best_config = sorted_results[0]
            st.markdown("#### 🎯 推荐最佳配置")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最佳配置", best_config[0])
            with col2:
                st.metric("最低损失", f"{best_config[1]['final_loss']:.4f}")
            with col3:
                st.metric("收敛速度", f"{best_config[1]['convergence_epoch']} epochs")

            # 参数建议
            st.markdown("#### 💡 参数优化建议")

            if comp_type == "学习率影响分析":
                lrs = [float(name.split("lr")[1]) for name in results.keys()]
                losses = [result["final_loss"] for result in results.values()]
                best_idx = np.argmin(losses)

                if best_idx == 0:
                    suggestion = "💡 建议尝试更小的学习率，可能还有改进空间"
                elif best_idx == len(lrs) - 1:
                    suggestion = "💡 建议尝试更大的学习率，当前最大值效果最好"
                else:
                    suggestion = (
                        f"💡 当前最优学习率 {lrs[best_idx]:.4f} 附近可以进一步细化搜索"
                    )

                st.info(suggestion)

            elif comp_type == "网络深度对比":
                depths = [int(name.split("layers")[0]) for name in results.keys()]
                losses = [result["final_loss"] for result in results.values()]
                best_idx = np.argmin(losses)

                if best_idx == 0:
                    suggestion = (
                        "💡 最浅的网络效果最好，说明当前问题可能不需要太深的网络"
                    )
                elif best_idx == len(depths) - 1:
                    suggestion = "💡 最深的网络效果最好，可以考虑继续增加深度"
                else:
                    suggestion = f"💡 {depths[best_idx]}层效果最佳，可以在这个附近微调"

                st.info(suggestion)


# 模拟训练函数
def simulate_training(**kwargs):
    """模拟训练过程，返回结果"""
    import time
    import random

    # 模拟训练时间
    training_time = random.uniform(0.5, 2.0)
    time.sleep(0.01)  # 模拟计算延迟

    # 根据参数生成模拟结果
    comparison_type = kwargs.get("comparison_type", "")

    if comparison_type == "优化器性能对比":
        optimizer = kwargs.get("optimizer", "SGD")
        lr = kwargs.get("learning_rate", 0.001)
        epochs = kwargs.get("epochs", 100)

        # 不同优化器的特性
        base_loss = 2.0
        optimizer_factor = {
            "SGD": 1.0,
            "Adam": 0.7,
            "RMSprop": 0.8,
            "Momentum": 0.75,
            "AdaGrad": 0.9,
            "Nesterov": 0.72,
        }
        lr_factor = min(lr / 0.01, 1.0) * 2  # 学习率影响

        final_loss = base_loss * optimizer_factor.get(
            optimizer, 1.0
        ) * lr_factor + random.uniform(-0.1, 0.1)
        final_loss = max(0.1, final_loss)  # 确保损失为正

    elif comparison_type == "学习率影响分析":
        lr = kwargs.get("learning_rate", 0.001)
        epochs = kwargs.get("epochs", 100)

        # 学习率对损失的影响（U型曲线）
        optimal_lr = 0.01
        lr_factor = 1 + abs(np.log(lr / optimal_lr))
        final_loss = 0.5 + lr_factor + random.uniform(-0.2, 0.2)

    elif comparison_type == "网络深度对比":
        depth = kwargs.get("depth", 4)
        hidden_dim = kwargs.get("hidden_dim", 64)
        lr = kwargs.get("learning_rate", 0.01)
        epochs = kwargs.get("epochs", 100)

        # 深度对损失的影响
        if depth <= 4:
            depth_factor = 1.0 + (4 - depth) * 0.2  # 欠拟合
        else:
            depth_factor = 0.8 + (depth - 4) * 0.05  # 过拟合

        final_loss = 0.3 + depth_factor + random.uniform(-0.1, 0.1)

    elif comparison_type == "激活函数性能":
        activation = kwargs.get("activation", "ReLU")
        lr = kwargs.get("learning_rate", 0.01)
        epochs = kwargs.get("epochs", 100)

        # 不同激活函数的性能
        activation_factor = {
            "ReLU": 0.8,
            "Leaky ReLU": 0.75,
            "ELU": 0.7,
            "Swish": 0.65,
            "GELU": 0.68,
            "Tanh": 0.9,
            "Sigmoid": 1.2,
        }

        final_loss = (
            0.4 + activation_factor.get(activation, 1.0) + random.uniform(-0.1, 0.1)
        )

    elif comparison_type == "正则化效果对比":
        dropout = kwargs.get("dropout", 0.0)
        l2_reg = kwargs.get("l2_reg", 0.0)
        lr = kwargs.get("learning_rate", 0.01)
        epochs = kwargs.get("epochs", 100)

        # 正则化对损失的影响
        if dropout == 0.0 and l2_reg == 0.0:
            reg_factor = 1.5  # 无正则化，过拟合
        else:
            reg_factor = 0.8 + dropout * 0.5 + l2_reg * 0.3

        final_loss = 0.6 + reg_factor + random.uniform(-0.1, 0.1)

    else:
        # 默认情况
        final_loss = 1.0 + random.uniform(-0.2, 0.2)

    # 生成损失曲线
    epochs = kwargs.get("epochs", 100)
    losses = []
    current_loss = 2.0

    for epoch in range(epochs):
        # 模拟损失下降过程
        decay_rate = 0.95 + random.uniform(-0.05, 0.05)
        noise = random.uniform(-0.02, 0.02)
        current_loss = max(final_loss, current_loss * decay_rate + noise)
        losses.append(current_loss)

    # 计算收敛epoch（损失降低到最终损失的1.1倍）
    convergence_threshold = final_loss * 1.1
    convergence_epoch = next(
        (i for i, loss in enumerate(losses) if loss <= convergence_threshold),
        epochs - 1,
    )

    # 生成准确率
    final_accuracy = max(
        0.5, min(0.95, 1.0 - final_loss / 2.0 + random.uniform(-0.05, 0.05))
    )

    return {
        "losses": losses,
        "final_loss": final_loss,
        "convergence_epoch": convergence_epoch,
        "final_accuracy": final_accuracy,
        "training_time": training_time,
    }


st.markdown("---")
st.markdown(
    "© 2025 "
    + (
        "神经网络数学原理探索器 | 深度学习数学教学工具"
        if CHINESE_SUPPORTED
        else "Neural Network Mathematics Explorer | Deep Learning Mathematics Teaching Tool"
    )
)
