"""
神经网络交互实验室模块
Interactive Lab Module for Neural Network Experiments

v2.2.0 新增：
- 统一稳定性检测
- 激活函数饱和检测
- 梯度消失自动诊断
- 超参数对比分析
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from simple_latex import display_latex
from utils.visualization.chart_optimizer import ChartOptimizer, cached_plotly_chart
from utils.numerical_stability_checker import StabilityChecker


def interactive_lab_tab(CHINESE_SUPPORTED):
    """交互实验室标签页内容"""

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
        key="interactive_lab_experiment_type",
    )

    if experiment_type == "CNN特征图可视化":
        _cnn_feature_visualization(CHINESE_SUPPORTED)
    elif experiment_type == "GNN节点分类演示":
        _gnn_node_classification(CHINESE_SUPPORTED)
    elif experiment_type == "激活函数对比":
        _activation_comparison(CHINESE_SUPPORTED)
    elif experiment_type == "优化器轨迹可视化":
        _optimizer_trajectory(CHINESE_SUPPORTED)
    elif experiment_type == "损失函数3D地形图":
        _loss_landscape_3d(CHINESE_SUPPORTED)
    else:  # 批量参数对比
        _batch_parameter_comparison(CHINESE_SUPPORTED)


def _cnn_feature_visualization(CHINESE_SUPPORTED):
    """CNN特征图可视化"""
    st.markdown("### CNN卷积特征图实时可视化")

    # 图像输入选项
    st.markdown("#### 📁 图像输入方式")
    input_method = st.radio(
        "选择输入方式", ["上传真实图像", "使用示例图像"], key="cnn_input_method"
    )

    input_image = None
    original_size = None

    if input_method == "上传真实图像":
        uploaded_file = st.file_uploader(
            "上传图像 (支持 JPG, PNG, GIF)",
            type=["jpg", "jpeg", "png", "gif"],
            help="上传你自己的图像来查看CNN处理效果",
            key="cnn_upload",
        )

        if uploaded_file is not None:
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
                target_size = st.slider(
                    "处理尺寸", 32, 256, 64, step=32, key="cnn_target_size"
                )
                image = image.resize((target_size, target_size))

                # 转换为numpy数组
                input_image = np.array(image).astype(float)

                # 归一化到[0, 1]
                input_image = input_image / 255.0

                # 显示输入图像
                st.markdown("**输入图像**")
                fig = px.imshow(input_image, color_continuous_scale="gray")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch", key="cnn_input_image")

            except Exception as e:
                st.error(f"图像加载失败: {str(e)}")
                input_image = None

    else:  # 使用示例图像
        image_size = st.slider("图像尺寸", 16, 64, 32, step=8, key="cnn_example_size")

        pattern_type = st.selectbox(
            "选择示例图案",
            ["随机噪声", "棋盘格", "圆形", "对角线", "梯度"],
            key="cnn_pattern",
        )

        # 生成不同类型的示例图像
        if pattern_type == "随机噪声":
            input_image = np.random.rand(image_size, image_size)
        elif pattern_type == "棋盘格":
            input_image = np.zeros((image_size, image_size))
            input_image[::2, ::2] = 1
            input_image[1::2, 1::2] = 1
        elif pattern_type == "圆形":
            input_image = np.zeros((image_size, image_size))
            center = image_size // 2
            radius = image_size // 3
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
            input_image[mask] = 1
        elif pattern_type == "对角线":
            input_image = np.eye(image_size)
        else:  # 梯度
            input_image = np.linspace(0, 1, image_size)
            input_image = np.tile(input_image, (image_size, 1))

        # 显示输入图像
        st.markdown("**输入图像**")
        fig = px.imshow(input_image, color_continuous_scale="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch", key="cnn_example_image")

    if input_image is not None:
        # 卷积核选择
        st.markdown("---")
        st.markdown("### 🎨 卷积核选择")

        col1, col2, col3 = st.columns(3)

        with col1:
            kernel_type = st.selectbox(
                "卷积核类型",
                ["边缘检测", "高斯模糊", "锐化", "浮雕", "自定义"],
                key="cnn_kernel_type",
            )

        with col2:
            stride = st.slider("步长 (Stride)", 1, 3, 1, key="cnn_stride")

        with col3:
            padding = st.slider("填充 (Padding)", 0, 3, 0, key="cnn_padding")

        # 定义卷积核
        kernels_dict = {
            "边缘检测": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "高斯模糊": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "浮雕": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        }

        if kernel_type == "自定义":
            st.markdown("**自定义卷积核 (3×3)**")
            k_cols = st.columns(3)
            kernel = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    with k_cols[j]:
                        kernel[i, j] = st.number_input(
                            f"[{i},{j}]",
                            value=0.0 if i != 1 or j != 1 else 1.0,
                            step=0.1,
                            key=f"cnn_kernel_{i}_{j}",
                        )
        else:
            kernel = kernels_dict[kernel_type]

        # 显示卷积核
        st.markdown("**卷积核矩阵**")
        st.dataframe(pd.DataFrame(kernel.round(3)), width=250)

        # 执行卷积
        st.markdown("---")
        st.markdown("### 🔄 卷积操作")

        # 添加padding
        if padding > 0:
            padded_input = np.pad(
                input_image, padding, mode="constant", constant_values=0
            )
        else:
            padded_input = input_image

        # 执行卷积
        feature_map = signal.convolve2d(padded_input, kernel, mode="valid")

        # 降采样（如果stride > 1）
        if stride > 1:
            feature_map = feature_map[::stride, ::stride]

        # 激活函数
        activation = st.selectbox(
            "激活函数", ["None", "ReLU", "Sigmoid", "Tanh"], key="cnn_activation"
        )

        if activation == "ReLU":
            feature_map = np.maximum(0, feature_map)
        elif activation == "Sigmoid":
            feature_map = 1 / (1 + np.exp(-np.clip(feature_map, -10, 10)))
        elif activation == "Tanh":
            feature_map = np.tanh(feature_map)

        # 显示结果
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**输入图像**")
            fig = px.imshow(input_image, color_continuous_scale="gray")
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch", key="cnn_feature_input")

        with col2:
            st.markdown("**输出特征图**")
            fig = px.imshow(feature_map, color_continuous_scale="viridis")
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch", key="cnn_feature_output")

        # 特征图统计
        st.markdown("**特征图统计**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最小值", f"{feature_map.min():.3f}")
        with col2:
            st.metric("最大值", f"{feature_map.max():.3f}")
        with col3:
            st.metric("均值", f"{feature_map.mean():.3f}")
        with col4:
            st.metric("标准差", f"{feature_map.std():.3f}")

        # 输出尺寸计算
        st.markdown("---")
        st.markdown("### 📏 输出尺寸计算")

        input_h, input_w = input_image.shape
        kernel_h, kernel_w = kernel.shape

        output_h = (input_h + 2 * padding - kernel_h) // stride + 1
        output_w = (input_w + 2 * padding - kernel_w) // stride + 1

        actual_h, actual_w = feature_map.shape

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
            **输入参数**:
            - 输入尺寸: {input_h} × {input_w}
            - 卷积核: {kernel_h} × {kernel_w}
            - 填充: {padding}
            - 步长: {stride}
            """
            )

        with col2:
            st.markdown(
                f"""
            **输出尺寸**:
            - 理论计算: {output_h} × {output_w}
            - 实际输出: {actual_h} × {actual_w}
            """
            )

        display_latex(
            r"H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1"
        )


def _gnn_node_classification(CHINESE_SUPPORTED):
    """GNN节点分类演示"""
    st.markdown("### 🕸️ GNN图节点分类演示")

    st.markdown(
        """
    演示图神经网络如何通过消息传递机制进行节点分类。
    """
    )

    # 图参数
    col1, col2 = st.columns(2)
    with col1:
        num_nodes = st.slider("节点数量", 5, 20, 10, key="gnn_nodes")
    with col2:
        edge_prob = st.slider("边连接概率", 0.1, 0.9, 0.3, key="gnn_edge_prob")

    # 生成随机图
    np.random.seed(42)
    adj_matrix = (np.random.rand(num_nodes, num_nodes) < edge_prob).astype(float)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
    np.fill_diagonal(adj_matrix, 0)  # 去除自环

    # 随机节点特征
    node_features = np.random.randn(num_nodes, 3)

    # 节点标签（随机分类）
    node_labels = np.random.randint(0, 3, num_nodes)
    label_colors = ["red", "green", "blue"]

    # 计算节点位置（使用力导向布局）
    from scipy.sparse.csgraph import shortest_path

    # 简单的圆形布局
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)

    # 可视化图
    st.markdown("#### 📊 图结构可视化")

    fig = go.Figure()

    # 绘制边（向量化：用numpy提取所有边坐标，单次add_trace）
    edge_mask = np.triu(adj_matrix > 0, k=1)
    edge_indices = np.argwhere(edge_mask)
    if len(edge_indices) > 0:
        edge_x = np.concatenate([pos_x[edge_indices[:, 0]], pos_x[edge_indices[:, 1]], [None] * len(edge_indices)])
        edge_y = np.concatenate([pos_y[edge_indices[:, 0]], pos_y[edge_indices[:, 1]], [None] * len(edge_indices))
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
                hoverinfo="none",
            )
        )

    # 绘制节点
    fig.add_trace(
        go.Scatter(
            x=pos_x,
            y=pos_y,
            mode="markers+text",
            marker=dict(
                size=20,
                color=[label_colors[label] for label in node_labels],
                line=dict(color="black", width=2),
            ),
            text=[f"{i}" for i in range(num_nodes)],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            showlegend=False,
            hovertext=[f"Node {i}<br>Class {node_labels[i]}" for i in range(num_nodes)],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="图结构 (节点颜色表示类别)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=500,
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch", key="gnn_graph")

    # 邻接矩阵
    st.markdown("#### 📐 邻接矩阵")
    fig_adj = px.imshow(adj_matrix, color_continuous_scale="Blues", aspect="auto")
    fig_adj.update_layout(height=400, title="邻接矩阵")
    st.plotly_chart(fig_adj, width="stretch", key="gnn_adj_matrix")

    # GNN消息传递
    st.markdown("#### 🔄 GNN消息传递")

    # 归一化邻接矩阵
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1  # 避免除零
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    A_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # 权重矩阵
    W = np.random.randn(3, 3) * 0.5

    # 一次消息传递
    H = node_features
    H_next = np.tanh(A_norm @ H @ W)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**初始节点特征**")
        st.dataframe(pd.DataFrame(H.round(3), columns=["f1", "f2", "f3"]))

    with col2:
        st.markdown("**传播后节点特征**")
        st.dataframe(pd.DataFrame(H_next.round(3), columns=["f1", "f2", "f3"]))

    st.markdown("**公式:**")
    display_latex(r"H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})")

    st.markdown(
        """
    其中：
    - $\\tilde{A}$: 归一化邻接矩阵
    - $H^{(l)}$: 第l层节点特征
    - $W^{(l)}$: 权重矩阵
    - $\\sigma$: 激活函数
    """
    )


def _activation_comparison(CHINESE_SUPPORTED):
    """激活函数对比"""
    st.markdown("### 🎯 激活函数交互式对比")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 参数设置")
        activations = st.multiselect(
            "选择激活函数",
            ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "ELU", "Swish", "GELU"],
            default=["ReLU", "Sigmoid", "Tanh"],
            key="act_functions",
        )

        x_range = st.slider("x范围", 1, 20, 5, key="act_x_range")
        num_points = st.slider("采样点数", 50, 500, 200, key="act_points")

    # 生成x值
    x = np.linspace(-x_range, x_range, num_points)

    # 定义激活函数
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def tanh(x):
        return np.tanh(x)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def swish(x):
        return x * sigmoid(x)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    activation_funcs = {
        "ReLU": relu,
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "Leaky ReLU": leaky_relu,
        "ELU": elu,
        "Swish": swish,
        "GELU": gelu,
    }

    with col2:
        st.markdown("#### 激活函数图像")

        fig = go.Figure()

        for act_name in activations:
            if act_name in activation_funcs:
                y = activation_funcs[act_name](x)
                fig.add_trace(go.Scatter(x=x, y=y, name=act_name, mode="lines"))

        fig = ChartOptimizer.optimize_figure(fig)

        # 使用缓存渲染，缓存键基于激活函数和x范围
        cache_data = {
            "type": "activation_functions",
            "activations": sorted(activations),
            "x_range": (x.min(), x.max()),
            "num_points": len(x),
        }
        cached_plotly_chart(
            fig, cache_key_data=cache_data, width="stretch", key="activation_functions"
        )

    # 导数对比
    st.markdown("---")
    st.markdown("#### 📉 激活函数导数对比")

    def numerical_derivative(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)

    fig = go.Figure()

    for act_name in activations:
        if act_name in activation_funcs:
            f = activation_funcs[act_name]
            dy = numerical_derivative(f, x)
            fig.add_trace(go.Scatter(x=x, y=dy, name=f"{act_name}'", mode="lines"))

    fig.update_layout(
        xaxis_title="x", yaxis_title="f'(x)", height=400, hovermode="x unified"
    )
    st.plotly_chart(fig, width="stretch", key="activation_derivatives")

    # ==================== 激活函数饱和检测 ====================
    st.markdown("---")
    st.markdown("#### 🔬 激活函数饱和分析")

    stability_issues = []

    for act_name in activations:
        if act_name in activation_funcs:
            f = activation_funcs[act_name]
            y = f(x)
            dy = numerical_derivative(f, x)

            # 检查饱和区域（导数接近0）
            saturated_ratio = np.sum(np.abs(dy) < 0.01) / len(dy)

            if saturated_ratio > 0.5:
                stability_issues.append(
                    {
                        "status": "warning",
                        "type": f"{act_name}饱和严重",
                        "value": f"{saturated_ratio*100:.1f}%区域饱和",
                        "threshold": "> 50%",
                        "icon": "🟡",
                        "severity": "medium",
                        "details": {
                            "激活函数": act_name,
                            "饱和比例": f"{saturated_ratio*100:.1f}%",
                            "最大导数": f"{np.max(np.abs(dy)):.4f}",
                            "平均导数": f"{np.mean(np.abs(dy)):.4f}",
                        },
                        "solution": [
                            f"避免使用{act_name}在深层网络",
                            "使用ReLU或其变体（Leaky ReLU, ELU）",
                            "使用GELU（Transformer默认）",
                            "添加BatchNorm/LayerNorm",
                        ],
                        "explanation": f"{act_name}在大部分区域导数接近0，导致梯度消失",
                    }
                )
            elif saturated_ratio > 0.3:
                stability_issues.append(
                    {
                        "status": "warning",
                        "type": f"{act_name}部分饱和",
                        "value": f"{saturated_ratio*100:.1f}%区域",
                        "threshold": "> 30%",
                        "icon": "🟡",
                        "severity": "low",
                        "details": {
                            "激活函数": act_name,
                            "饱和比例": f"{saturated_ratio*100:.1f}%",
                        },
                        "solution": ["监控训练时的激活值分布", "考虑使用其他激活函数"],
                        "explanation": f"{act_name}有一定比例的饱和区域",
                    }
                )

    if stability_issues:
        StabilityChecker.display_issues(stability_issues, title="🔬 激活函数饱和诊断")
    else:
        st.success("✅ 选择的激活函数在当前范围内饱和程度较低")

    # 梯度消失分析
    st.markdown("---")
    st.markdown("#### ⚠️ 梯度传播分析")

    col1, col2 = st.columns(2)

    with col1:
        depth = st.slider("网络深度", 5, 50, 20, key="act_depth")
        input_val = st.slider("输入值", -3.0, 3.0, 1.0, key="act_input")

    with col2:
        test_activation = st.selectbox(
            "测试激活函数", list(activation_funcs.keys()), key="act_test"
        )

    # 模拟梯度传播
    values = [input_val]
    gradients = [1.0]

    for i in range(depth):
        val = values[-1]
        grad = gradients[-1]

        # 应用激活函数
        activated = activation_funcs[test_activation](val)

        # 计算导数
        derivative = numerical_derivative(activation_funcs[test_activation], val)

        values.append(activated)
        gradients.append(grad * derivative)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("激活值传播", "梯度反向传播"))

    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))), y=values, mode="lines+markers", name="激活值"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(gradients))),
            y=gradients,
            mode="lines+markers",
            name="梯度",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="层数", row=1, col=1)
    fig.update_xaxes(title_text="层数", row=1, col=2)
    fig.update_yaxes(title_text="值", row=1, col=1)
    fig.update_yaxes(title_text="梯度", row=1, col=2, type="log")

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width="stretch", key="gradient_propagation")

    final_grad = abs(gradients[-1])

    if final_grad < 1e-5:
        st.error(f"⚠️ **梯度消失！** 最终梯度: {final_grad:.2e}")
    elif final_grad > 1e5:
        st.error(f"⚠️ **梯度爆炸！** 最终梯度: {final_grad:.2e}")
    else:
        st.success(f"✅ **梯度稳定。** 最终梯度: {final_grad:.2e}")


def _optimizer_trajectory(CHINESE_SUPPORTED):
    """优化器轨迹可视化"""
    st.markdown("### 🎯 优化器轨迹可视化")

    st.markdown("比较不同优化器在2D损失函数上的优化轨迹。")

    # 参数设置
    col1, col2, col3 = st.columns(3)

    with col1:
        loss_type = st.selectbox(
            "损失函数", ["Bowl", "Rosenbrock", "Beale"], key="opt_loss"
        )
    with col2:
        optimizers = st.multiselect(
            "优化器",
            ["SGD", "Momentum", "Adam", "RMSprop"],
            default=["SGD", "Adam"],
            key="opt_optimizers",
        )
    with col3:
        learning_rate = st.slider("学习率", 0.001, 0.5, 0.1, key="opt_lr")

    # 定义损失函数
    def bowl(x, y):
        return x**2 + y**2

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def beale(x, y):
        return (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y**2) ** 2
            + (2.625 - x + x * y**3) ** 2
        )

    loss_funcs = {"Bowl": bowl, "Rosenbrock": rosenbrock, "Beale": beale}

    loss_func = loss_funcs[loss_type]

    # 生成等高线数据
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_func(X, Y)

    # 优化器实现（简化版）
    def optimize_sgd(start, lr, steps=50):
        trajectory = [start]
        pos = np.array(start, dtype=float)

        for _ in range(steps):
            grad = np.array(
                [
                    (
                        loss_func(pos[0] + 1e-5, pos[1])
                        - loss_func(pos[0] - 1e-5, pos[1])
                    )
                    / (2e-5),
                    (
                        loss_func(pos[0], pos[1] + 1e-5)
                        - loss_func(pos[0], pos[1] - 1e-5)
                    )
                    / (2e-5),
                ]
            )
            pos = pos - lr * grad
            trajectory.append(pos.copy())

        return np.array(trajectory)

    def optimize_momentum(start, lr, steps=50, beta=0.9):
        trajectory = [start]
        pos = np.array(start, dtype=float)
        velocity = np.zeros(2)

        for _ in range(steps):
            grad = np.array(
                [
                    (
                        loss_func(pos[0] + 1e-5, pos[1])
                        - loss_func(pos[0] - 1e-5, pos[1])
                    )
                    / (2e-5),
                    (
                        loss_func(pos[0], pos[1] + 1e-5)
                        - loss_func(pos[0], pos[1] - 1e-5)
                    )
                    / (2e-5),
                ]
            )
            velocity = beta * velocity + (1 - beta) * grad
            pos = pos - lr * velocity
            trajectory.append(pos.copy())

        return np.array(trajectory)

    def optimize_adam(start, lr, steps=50, beta1=0.9, beta2=0.999, epsilon=1e-8):
        trajectory = [start]
        pos = np.array(start, dtype=float)
        m = np.zeros(2)
        v = np.zeros(2)

        for t in range(1, steps + 1):
            grad = np.array(
                [
                    (
                        loss_func(pos[0] + 1e-5, pos[1])
                        - loss_func(pos[0] - 1e-5, pos[1])
                    )
                    / (2e-5),
                    (
                        loss_func(pos[0], pos[1] + 1e-5)
                        - loss_func(pos[0], pos[1] - 1e-5)
                    )
                    / (2e-5),
                ]
            )
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            pos = pos - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            trajectory.append(pos.copy())

        return np.array(trajectory)

    def optimize_rmsprop(start, lr, steps=50, beta=0.9, epsilon=1e-8):
        trajectory = [start]
        pos = np.array(start, dtype=float)
        cache = np.zeros(2)

        for _ in range(steps):
            grad = np.array(
                [
                    (
                        loss_func(pos[0] + 1e-5, pos[1])
                        - loss_func(pos[0] - 1e-5, pos[1])
                    )
                    / (2e-5),
                    (
                        loss_func(pos[0], pos[1] + 1e-5)
                        - loss_func(pos[0], pos[1] - 1e-5)
                    )
                    / (2e-5),
                ]
            )
            cache = beta * cache + (1 - beta) * grad**2
            pos = pos - lr * grad / (np.sqrt(cache) + epsilon)
            trajectory.append(pos.copy())

        return np.array(trajectory)

    optimizer_funcs = {
        "SGD": optimize_sgd,
        "Momentum": optimize_momentum,
        "Adam": optimize_adam,
        "RMSprop": optimize_rmsprop,
    }

    # 起始点
    start_point = [-1.5, 1.5]

    # 可视化
    fig = go.Figure()

    # 等高线
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=np.log(Z + 1),  # log scale for better visualization
            colorscale="Viridis",
            showscale=False,
            opacity=0.6,
            contours=dict(showlabels=False),
        )
    )

    # 优化轨迹
    colors = ["red", "blue", "green", "orange"]

    for i, opt_name in enumerate(optimizers):
        if opt_name in optimizer_funcs:
            trajectory = optimizer_funcs[opt_name](start_point, learning_rate)

            fig.add_trace(
                go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode="lines+markers",
                    name=opt_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                )
            )

    # 起始点
    fig.add_trace(
        go.Scatter(
            x=[start_point[0]],
            y=[start_point[1]],
            mode="markers",
            name="起始点",
            marker=dict(size=15, color="black", symbol="star"),
        )
    )

    fig.update_layout(
        title=f"{loss_type} 损失函数上的优化轨迹",
        xaxis_title="x",
        yaxis_title="y",
        height=600,
    )

    st.plotly_chart(fig, width="stretch", key="optimization_trajectory")


def _loss_landscape_3d(CHINESE_SUPPORTED):
    """损失函数3D地形图"""
    st.markdown("### 🗻 损失函数3D地形图")

    st.markdown("探索不同损失函数的三维地形。")

    # 参数设置
    col1, col2, col3 = st.columns(3)

    with col1:
        loss_type = st.selectbox(
            "损失函数",
            ["Sphere", "Rosenbrock", "Himmelblau", "Rastrigin"],
            key="loss_type",
        )
    with col2:
        resolution = st.slider("分辨率", 20, 100, 50, key="loss_resolution")
    with col3:
        scale = st.slider("范围", 1, 10, 5, key="loss_scale")

    # 定义损失函数
    def sphere(x, y):
        return x**2 + y**2

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    def himmelblau(x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def rastrigin(x, y):
        return (
            20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)
        )

    loss_funcs = {
        "Sphere": sphere,
        "Rosenbrock": rosenbrock,
        "Himmelblau": himmelblau,
        "Rastrigin": rastrigin,
    }

    loss_func = loss_funcs[loss_type]

    # 生成网格
    x = np.linspace(-scale, scale, resolution)
    y = np.linspace(-scale, scale, resolution)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)

    # 3D曲面图
    st.markdown("#### 🌄 3D曲面图")

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale="Viridis")])

    fig.update_layout(
        title=f"{loss_type} 损失函数3D地形",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=600,
    )

    st.plotly_chart(fig, width="stretch", key="loss_3d_surface")

    # 等高线图
    st.markdown("#### 📏 等高线图")

    fig2 = go.Figure(
        data=go.Contour(
            x=x, y=y, z=Z, colorscale="Viridis", contours=dict(showlabels=True)
        )
    )

    fig2.update_layout(
        title=f"{loss_type} 损失函数等高线",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
    )

    st.plotly_chart(fig2, width="stretch", key="loss_contour")


def _batch_parameter_comparison(CHINESE_SUPPORTED):
    """批量参数对比"""
    st.markdown("### 🚀 批量参数对比实验")

    st.markdown(
        """
    同时对比多组超参数对模型性能的影响。
    """
    )

    # 实验设置
    st.markdown("#### ⚙️ 实验设置")

    col1, col2 = st.columns(2)

    with col1:
        param_type = st.selectbox(
            "对比参数",
            ["学习率", "批次大小", "网络深度", "隐藏层大小"],
            key="batch_param_type",
        )

    with col2:
        num_experiments = st.slider("实验数量", 3, 8, 5, key="batch_num_exp")

    # 根据参数类型生成不同的值
    if param_type == "学习率":
        param_values = np.logspace(-4, -1, num_experiments)
        param_name = "Learning Rate"
    elif param_type == "批次大小":
        param_values = [2**i for i in range(4, 4 + num_experiments)]
        param_name = "Batch Size"
    elif param_type == "网络深度":
        param_values = list(range(2, 2 + num_experiments))
        param_name = "Depth"
    else:  # 隐藏层大小
        param_values = [2 ** (i + 4) for i in range(num_experiments)]
        param_name = "Hidden Size"

    # 模拟训练结果
    np.random.seed(42)
    epochs = 50

    results = {}

    for i, val in enumerate(param_values):
        # 模拟训练曲线
        if param_type == "学习率":
            # 学习率影响收敛速度和稳定性
            if val < 0.001:
                train_loss = (
                    2.0 * np.exp(-0.02 * np.arange(epochs))
                    + np.random.randn(epochs) * 0.05
                )
            elif val < 0.01:
                train_loss = (
                    2.0 * np.exp(-0.05 * np.arange(epochs))
                    + np.random.randn(epochs) * 0.03
                )
            else:
                train_loss = (
                    2.0 * np.exp(-0.03 * np.arange(epochs))
                    + np.random.randn(epochs) * 0.1
                )
        else:
            # 其他参数的简化模拟
            decay_rate = 0.03 + np.random.rand() * 0.02
            train_loss = (
                2.0 * np.exp(-decay_rate * np.arange(epochs))
                + np.random.randn(epochs) * 0.05
            )

        train_loss = np.maximum(train_loss, 0.1)  # 确保非负
        results[val] = train_loss

    # 可视化训练曲线
    st.markdown("---")
    st.markdown("#### 📊 训练曲线对比")

    fig = go.Figure()

    for val, loss in results.items():
        if param_type == "学习率":
            label = f"LR={val:.4f}"
        elif param_type == "批次大小":
            label = f"BS={int(val)}"
        elif param_type == "网络深度":
            label = f"Depth={int(val)}"
        else:
            label = f"Hidden={int(val)}"

        fig.add_trace(
            go.Scatter(x=list(range(epochs)), y=loss, mode="lines", name=label)
        )

    fig.update_layout(
        title=f"{param_type}对训练损失的影响",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=500,
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch", key="batch_comparison")

    # 最终性能对比
    st.markdown("#### 🏆 最终性能对比")

    final_losses = {val: loss[-1] for val, loss in results.items()}
    best_param = min(final_losses, key=final_losses.get)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=[
                        f"{val:.4f}" if isinstance(val, float) else str(int(val))
                        for val in final_losses.keys()
                    ],
                    y=list(final_losses.values()),
                    marker_color=[
                        "green" if val == best_param else "lightblue"
                        for val in final_losses.keys()
                    ],
                )
            ]
        )

        fig2.update_layout(
            title=f"最终损失对比 ({param_type})",
            xaxis_title=param_name,
            yaxis_title="Final Loss",
            height=400,
        )

        st.plotly_chart(fig2, width="stretch", key="final_performance")

    with col2:
        st.markdown("**最佳参数**")
        if param_type == "学习率":
            st.metric("最佳学习率", f"{best_param:.4f}")
        else:
            st.metric(f"最佳{param_type}", f"{int(best_param)}")

        st.metric("最终损失", f"{final_losses[best_param]:.4f}")

        improvement = (
            (max(final_losses.values()) - min(final_losses.values()))
            / max(final_losses.values())
            * 100
        )
        st.metric("性能提升", f"{improvement:.1f}%")

    # 统计分析
    st.markdown("---")
    st.markdown("#### 📈 统计分析")

    # 创建统计表格
    stats_data = []
    for val, loss in results.items():
        stats_data.append(
            {
                param_name: f"{val:.4f}" if isinstance(val, float) else int(val),
                "最终损失": f"{loss[-1]:.4f}",
                "最小损失": f"{loss.min():.4f}",
                "收敛速度": f"{np.where(loss < loss[0] * 0.5)[0][0] if any(loss < loss[0] * 0.5) else epochs}",
            }
        )

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, width=800)


if __name__ == "__main__":
    # 独立运行时的测试
    interactive_lab_tab(True)
