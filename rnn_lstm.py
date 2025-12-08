"""
RNN/LSTM 时序神经网络数学原理模块
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from simple_latex import display_latex


def rnn_lstm_tab(CHINESE_SUPPORTED):
    """RNN/LSTM 标签页内容"""

    st.header("🔄 RNN/LSTM 时序神经网络数学原理")

    # ==========================================
    # 第一部分：RNN基本概念
    # ==========================================
    st.markdown("### 🧠 RNN：有记忆的神经网络")

    with st.expander("💡 核心概念", expanded=True):
        st.markdown(
            """
        **RNN就像是有记忆的人：**
        
        1. 📖 **记住过去** - 每个时间步都保留之前的信息
        2. 🔗 **信息传递** - 通过隐藏状态连接时间序列
        3. 🔄 **循环计算** - 相同的权重在每个时间步重复使用
        4. 📊 **序列处理** - 专门处理变长序列数据
        
        **关键参数：**
        - **序列长度**：处理的时间步数
        - **隐藏层大小**：记忆容量
        - **RNN类型**：Simple RNN、LSTM、GRU
        """
        )

    # ==========================================
    # 第二部分：数学原理展示
    # ==========================================
    st.markdown("### 📐 RNN 数学原理")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**RNN 前向传播公式**")
        display_latex("h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)")
        display_latex("y_t = W_{hy} h_t + b_y")

        st.markdown("**参数说明：**")
        st.markdown("- $h_t$: 时间步t的隐藏状态")
        st.markdown("- $x_t$: 时间步t的输入")
        st.markdown("- $y_t$: 时间步t的输出")
        st.markdown("- $W_{hh}$: 隐藏状态到隐藏状态的权重")
        st.markdown("- $W_{xh}$: 输入到隐藏状态的权重")
        st.markdown("- $W_{hy}$: 隐藏状态到输出的权重")

        st.markdown("**反向传播 Through Time (BPTT)**")
        display_latex(
            "\\frac{\\partial L}{\\partial h_t} = \\frac{\\partial L_t}{\\partial h_t} + \\frac{\\partial L}{\\partial h_{t+1}} \\cdot \\frac{\\partial h_{t+1}}{\\partial h_t}"
        )

    with col2:
        # RNN计算演示
        st.markdown("**🔍 RNN 计算演示**")

        seq_len_demo = st.slider("演示序列长度", 3, 10, 5, key="rnn_seq_len")
        hidden_size_demo = st.slider("隐藏层大小", 2, 8, 4, key="rnn_hidden_size")

        # 随机初始化参数
        np.random.seed(42)
        W_hh = np.random.randn(hidden_size_demo, hidden_size_demo) * 0.1
        W_xh = np.random.randn(hidden_size_demo, 1) * 0.1  # 假设输入维度为1
        W_hy = np.random.randn(1, hidden_size_demo) * 0.1

        # 生成输入序列
        x_sequence = np.sin(np.linspace(0, 4 * np.pi, seq_len_demo))

        # 前向传播
        h_states = []
        y_outputs = []
        h_prev = np.zeros(hidden_size_demo)

        for t in range(seq_len_demo):
            x_t = np.array([[x_sequence[t]]])
            h_t = np.tanh(W_hh @ h_prev + W_xh @ x_t)
            y_t = W_hy @ h_t

            h_states.append(h_t.flatten())  # 确保1D
            y_outputs.append(y_t[0, 0])
            h_prev = h_t

        # 可视化
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["输入序列 & 隐藏状态", "输出序列"],
            vertical_spacing=0.1,
        )

        # 输入序列
        fig.add_trace(
            go.Scatter(
                x=list(range(seq_len_demo)),
                y=x_sequence,
                mode="lines+markers",
                name="输入序列",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # 隐藏状态（取第一个维度）
        fig.add_trace(
            go.Scatter(
                x=list(range(seq_len_demo)),
                y=[h[0] for h in h_states],
                mode="lines+markers",
                name="隐藏状态[0]",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        # 输出序列
        fig.add_trace(
            go.Scatter(
                x=list(range(seq_len_demo)),
                y=y_outputs,
                mode="lines+markers",
                name="输出序列",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, width="stretch")

        # 显示隐藏状态演化
        st.markdown("**隐藏状态演化矩阵**")
        h_matrix = np.array(h_states).T  # shape: (hidden_size_demo, seq_len_demo)
        fig = px.imshow(
            h_matrix,
            labels=dict(x="时间步", y="隐藏单元", color="激活值"),
            color_continuous_scale="RdBu",
            aspect="auto",
            title=f"隐藏状态演化 ({hidden_size_demo}×{seq_len_demo})",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    # ==========================================
    # 第三部分：LSTM门控机制
    # ==========================================
    st.markdown("---")
    st.markdown("### 🚪 LSTM 门控机制")

    with st.expander("🔐 LSTM 三大门控", expanded=False):
        st.markdown(
            """
        **LSTM就像是有三个门的房间：**
        
        1. **遗忘门 (Forget Gate)** - 决定丢弃哪些信息
        2. **输入门 (Input Gate)** - 决定存储哪些新信息  
        3. **输出门 (Output Gate)** - 决定输出哪些信息
        
        **细胞状态 (Cell State)** - 长期记忆的载体
        """
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**LSTM 数学公式**")

        st.markdown("**遗忘门**")
        display_latex("f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)")

        st.markdown("**输入门**")
        display_latex("i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)")
        display_latex("\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)")

        st.markdown("**细胞状态更新**")
        display_latex("C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t")

        st.markdown("**输出门**")
        display_latex("o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)")
        display_latex("h_t = o_t \\odot \\tanh(C_t)")

        st.markdown("**符号说明**")
        st.markdown("- $\\sigma$: sigmoid函数")
        st.markdown("- $\\odot$: 逐元素乘积")
        st.markdown("- $[h_{t-1}, x_t]$: 拼接向量")

    with col2:
        st.markdown("**🎮 LSTM 门控可视化**")

        # 简化的LSTM演示
        time_step = st.slider("选择时间步", 0, 4, 2, key="lstm_time_step")

        # 模拟门控值
        gate_values = {
            "遗忘门": [0.8, 0.3, 0.6, 0.9, 0.4],
            "输入门": [0.2, 0.7, 0.4, 0.1, 0.8],
            "输出门": [0.5, 0.6, 0.3, 0.7, 0.2],
        }

        # 当前时间步的门控值
        current_gates = {
            gate: values[time_step] for gate, values in gate_values.items()
        }

        # 可视化门控状态
        fig = go.Figure()

        gates = list(current_gates.keys())
        values = list(current_gates.values())
        colors = ["red", "green", "blue"]

        fig.add_trace(
            go.Bar(
                x=gates,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                marker_color=colors,
                name=f"时间步 {time_step}",
            )
        )

        fig.update_layout(
            title=f"LSTM 门控状态 (时间步 {time_step})",
            xaxis_title="门控类型",
            yaxis_title="激活值 (0-1)",
            yaxis=dict(range=[0, 1]),
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

        # 解释当前状态
        st.markdown("**📊 门控状态解释**")
        if current_gates["遗忘门"] > 0.5:
            st.markdown(
                f"🔴 **遗忘门 ({current_gates['遗忘门']:.2f})**: 保留大部分历史信息"
            )
        else:
            st.markdown(
                f"🔴 **遗忘门 ({current_gates['遗忘门']:.2f})**: 遗忘较多历史信息"
            )

        if current_gates["输入门"] > 0.5:
            st.markdown(
                f"🟢 **输入门 ({current_gates['输入门']:.2f})**: 接受较多新信息"
            )
        else:
            st.markdown(
                f"🟢 **输入门 ({current_gates['输入门']:.2f})**: 拒绝大部分新信息"
            )

        if current_gates["输出门"] > 0.5:
            st.markdown(
                f"🔵 **输出门 ({current_gates['输出门']:.2f})**: 输出较多内部状态"
            )
        else:
            st.markdown(
                f"🔵 **输出门 ({current_gates['输出门']:.2f})**: 输出较少内部状态"
            )

    # ==========================================
    # 第四部分：梯度消失/爆炸演示
    # ==========================================
    st.markdown("---")
    st.markdown("### 📉 梯度消失与爆炸")

    gradient_demo = st.selectbox(
        "选择演示类型",
        ["梯度消失", "梯度爆炸", "LSTM vs RNN 对比"],
        key="gradient_demo_type",
    )

    if gradient_demo == "梯度消失":
        st.markdown(
            """
        **梯度消失的原因：**
        - RNN中梯度通过时间反向传播
        - 每个时间步都要乘以权重矩阵
        - 如果权重值<1，梯度指数级衰减
        
        **数学直觉：**
        $$\\frac{\\partial h_T}{\\partial h_t} = \\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}}$$
        如果 $|\\frac{\\partial h_k}{\\partial h_{k-1}}| < 1$，则梯度趋向于0
        """
        )

        # 梯度消失演示
        time_steps = st.slider("时间步数", 5, 50, 20, key="vanishing_steps")
        weight_scale = st.slider("权重缩放", 0.1, 0.9, 0.5, key="vanishing_weight")

        gradients = []
        grad = 1.0
        for t in range(time_steps):
            grad = grad * weight_scale  # 简化模型：梯度每步乘以权重
            gradients.append(grad)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(time_steps)),
                y=gradients,
                mode="lines+markers",
                name=f"权重={weight_scale}",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title="梯度消失演示",
            xaxis_title="时间步",
            yaxis_title="梯度大小",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"**观察：** 经过{time_steps}步后，梯度从1.0衰减到{gradients[-1]:.6f}"
        )

    elif gradient_demo == "梯度爆炸":
        st.markdown(
            """
        **梯度爆炸的原因：**
        - 与梯度消失相反
        - 如果权重值>1，梯度指数级增长
        - 可能导致数值溢出和训练不稳定
        
        **解决方案：**
        - 梯度裁剪 (Gradient Clipping)
        - 权重初始化策略
        - 使用LSTM/GRU等改进结构
        """
        )

        # 梯度爆炸演示
        time_steps = st.slider("时间步数", 5, 20, 10, key="exploding_steps")
        weight_scale = st.slider("权重缩放", 1.1, 2.0, 1.5, key="exploding_weight")

        gradients = []
        grad = 1.0
        for t in range(time_steps):
            grad = grad * weight_scale
            gradients.append(grad)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(time_steps)),
                y=gradients,
                mode="lines+markers",
                name=f"权重={weight_scale}",
                line=dict(color="orange", width=2),
            )
        )

        fig.update_layout(
            title="梯度爆炸演示",
            xaxis_title="时间步",
            yaxis_title="梯度大小",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"**观察：** 经过{time_steps}步后，梯度从1.0增长到{gradients[-1]:.2f}"
        )

        # 梯度裁剪演示
        clip_threshold = st.slider("梯度裁剪阈值", 0.5, 5.0, 1.0, key="clip_threshold")
        clipped_gradients = [min(g, clip_threshold) for g in gradients]

        fig.add_trace(
            go.Scatter(
                x=list(range(time_steps)),
                y=clipped_gradients,
                mode="lines+markers",
                name=f"裁剪后(阈值={clip_threshold})",
                line=dict(color="green", width=2, dash="dash"),
            )
        )
        st.plotly_chart(fig, width="stretch")

    else:  # LSTM vs RNN 对比
        st.markdown(
            """
        **LSTM 如何解决梯度问题：**
        
        1. **细胞状态 (Cell State)** - 提供梯度高速公路
        2. **加法运算** - 而不是乘法，避免梯度衰减
        3. **门控机制** - 智能控制信息流动
        
        **RNN vs LSTM 梯度流动对比**
        """
        )

        # 对比演示
        time_steps = st.slider("时间步数", 10, 100, 50, key="compare_steps")

        # 模拟RNN梯度（指数衰减）
        rnn_gradients = []
        grad = 1.0
        for t in range(time_steps):
            grad = grad * 0.9  # RNN权重<1，梯度衰减
            rnn_gradients.append(grad)

        # 模拟LSTM梯度（相对稳定）
        lstm_gradients = []
        grad = 1.0
        for t in range(time_steps):
            grad = grad * (0.95 + 0.1 * np.random.random())  # LSTM更稳定
            lstm_gradients.append(grad)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(time_steps)),
                y=rnn_gradients,
                mode="lines",
                name="RNN梯度",
                line=dict(color="red", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(time_steps)),
                y=lstm_gradients,
                mode="lines",
                name="LSTM梯度",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="RNN vs LSTM 梯度稳定性对比",
            xaxis_title="时间步",
            yaxis_title="梯度大小",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        # 统计对比
        rnn_final = rnn_gradients[-1]
        lstm_final = lstm_gradients[-1]
        rnn_decay = rnn_final / rnn_gradients[0]
        lstm_decay = lstm_final / lstm_gradients[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("RNN最终梯度", f"{rnn_final:.6f}")
            st.metric("RNN衰减比", f"{rnn_decay:.6f}")
        with col2:
            st.metric("LSTM最终梯度", f"{lstm_final:.6f}")
            st.metric("LSTM衰减比", f"{lstm_decay:.6f}")

    # ==========================================
    # 第五部分：时间序列预测示例
    # ==========================================
    st.markdown("---")
    st.markdown("### 📈 时间序列预测交互示例")

    st.markdown("**🎮 尝试训练一个简单的RNN/LSTM来预测正弦波**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**📊 数据生成与参数设置**")

        # 数据参数
        data_length = st.slider("数据长度", 100, 500, 200, key="data_length")
        noise_level = st.slider("噪声水平", 0.0, 0.5, 0.1, key="noise_level")

        # 模型参数
        model_type = st.selectbox(
            "模型类型", ["Simple RNN", "LSTM", "GRU"], key="model_type"
        )
        hidden_size = st.slider("隐藏层大小", 8, 64, 32, key="model_hidden_size")
        learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, key="model_learning_rate")

        # 生成数据
        t = np.linspace(0, 4 * np.pi, data_length)
        clean_signal = np.sin(t)
        noise = np.random.normal(0, noise_level, data_length)
        noisy_signal = clean_signal + noise

        # 使用动态参数建议器
        from utils.parameter_suggester import get_suggested_params

        try:
            # 获取用户选择的序列长度
            sequence_length = st.session_state.get("lstm_sequence_length", 20)
        except:
            # 如果获取失败，使用动态建议
            suggested_params = get_suggested_params(
                "rnn",
                sequence_length=20,
                input_size=1,  # 单变量时间序列
                task_type="regression",
            )
            sequence_length = 20  # 保持默认值，但可以扩展
        X, y = [], []
        for i in range(len(noisy_signal) - sequence_length):
            X.append(noisy_signal[i : i + sequence_length])
            y.append(noisy_signal[i + sequence_length])

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        # 简单训练演示（模拟）
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if st.button("🚀 开始训练", key="train_button"):
            st.markdown("**训练中...**")

            # 使用动态性能预测模拟训练过程
            from utils.training import simulate_training

            # 获取用户选择的参数
            hidden_size = st.session_state.get("lstm_hidden", 256)
            num_layers = st.session_state.get("lstm_layers", 2)

            # 估算模型参数数量
            num_params = (
                4
                * (sequence_length * hidden_size + hidden_size * hidden_size)
                * num_layers
            )

            # 模拟训练过程
            training_result = simulate_training(
                epochs=50,
                model_type="RNN",
                num_params=num_params,
                num_classes=1,  # 回归任务
                dataset_size=len(X_train),
                learning_rate=0.001,
            )

            train_losses = training_result["train_loss"]
            val_losses = training_result["val_loss"]

            # 显示训练曲线
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(50)),
                    y=train_losses,
                    mode="lines",
                    name="训练损失",
                    line=dict(color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(50)),
                    y=val_losses,
                    mode="lines",
                    name="验证损失",
                    line=dict(color="red"),
                )
            )

            fig.update_layout(
                title="训练过程", xaxis_title="Epoch", yaxis_title="损失", height=300
            )
            st.plotly_chart(fig, width="stretch")

            # 模拟预测结果
            predictions = []
            current_seq = X_test[0].flatten()

            for _ in range(len(X_test)):
                # 简单的预测模拟（实际应该用训练好的模型）
                pred = np.mean(current_seq[-5:]) + 0.1 * np.random.random()
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred

            st.success("✅ 训练完成！")

    with col2:
        st.markdown("**📈 预测结果可视化**")

        # 显示原始数据
        fig = go.Figure()

        # 原始信号
        fig.add_trace(
            go.Scatter(
                x=t,
                y=clean_signal,
                mode="lines",
                name="真实信号",
                line=dict(color="blue", width=2),
                opacity=0.7,
            )
        )

        # 带噪声信号
        fig.add_trace(
            go.Scatter(
                x=t,
                y=noisy_signal,
                mode="lines",
                name="带噪声信号",
                line=dict(color="lightblue"),
                opacity=0.5,
            )
        )

        fig.update_layout(
            title="时间序列数据", xaxis_title="时间", yaxis_title="值", height=300
        )
        st.plotly_chart(fig, width="stretch")

        # 如果有预测结果，显示预测对比
        if "predictions" in locals():
            test_t = t[
                train_size
                + sequence_length : train_size
                + sequence_length
                + len(predictions)
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=test_t,
                    y=y_test,
                    mode="lines",
                    name="真实值",
                    line=dict(color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_t,
                    y=predictions,
                    mode="lines",
                    name="预测值",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig.update_layout(
                title="预测结果对比", xaxis_title="时间", yaxis_title="值", height=300
            )
            st.plotly_chart(fig, width="stretch")

            # 计算评估指标
            mse = np.mean((np.array(predictions) - y_test) ** 2)
            mae = np.mean(np.abs(np.array(predictions) - y_test))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("均方误差 (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")


if __name__ == "__main__":
    # 独立运行时的测试
    rnn_lstm_tab(True)
