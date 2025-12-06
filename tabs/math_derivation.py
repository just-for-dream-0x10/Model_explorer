"""
数学推导工具标签页
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from sympy import symbols, latex
from simple_latex import display_latex


def math_derivation_tab():
    """数学推导工具标签页"""
    st.header("📐 交互式数学推导工具")

    derivation_type = st.selectbox(
        "选择推导主题",
        ["卷积定理推导", "梯度下降优化", "反向传播链式法则", "图拉普拉斯矩阵"],
    )

    if derivation_type == "卷积定理推导":
        _convolution_theorem()
    elif derivation_type == "梯度下降优化":
        _gradient_descent()
    elif derivation_type == "反向传播链式法则":
        _backprop_chain_rule()
    elif derivation_type == "图拉普拉斯矩阵":
        _graph_laplacian()


def _convolution_theorem():
    """卷积定理推导"""
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
    display_latex("= \\mathcal{F}\\{f\\}(\\omega) \\cdot \\mathcal{F}\\{g\\}(\\omega)")
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


def _gradient_descent():
    """梯度下降优化"""
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
    fig.add_trace(go.Scatter(x=x, y=y, name="f(x) = x²", line=dict(color="lightgray")))
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


def _backprop_chain_rule():
    """反向传播链式法则"""
    st.markdown("### 反向传播链式法则推导")

    st.markdown("#### 链式法则：")
    st.markdown("对于复合函数 $y = f(g(x))$，有：")
    display_latex(r"\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}")

    st.markdown("#### 神经网络中的应用：")
    st.markdown("对于 $L$ 层网络，损失函数对第 $l$ 层参数的梯度：")
    display_latex(
        r"\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \ldots \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}"
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


def _graph_laplacian():
    """图拉普拉斯矩阵"""
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
    D = np.diag(np.sum(A, axis=1).A1)  # 修复：使用 .A1 转换为1D数组
    L = D - A

    # 归一化拉普拉斯
    try:
        D_sqrt = np.sqrt(D)
        D_inv_sqrt = np.linalg.inv(D_sqrt)
        L_sym = np.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    except np.linalg.LinAlgError:
        # 处理奇异矩阵情况（度为0的节点）
        D_sqrt = np.sqrt(D)
        D_inv_sqrt = np.zeros_like(D_sqrt)
        # 只对非零元素求逆
        for i in range(num_nodes):
            if D_sqrt[i, i] > 1e-10:
                D_inv_sqrt[i, i] = 1.0 / D_sqrt[i, i]
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
                np.array(L_sym).round(3),
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
