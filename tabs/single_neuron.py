"""
单神经元可视化模块 - Single Neuron Visualization
==================================================

通过单个神经元的视角，深入理解神经网络的工作原理。

核心概念：
1. 神经元是神经网络的基本计算单元
2. 前向传播：加权和 + 激活函数
3. 反向传播：链式法则计算梯度
4. 参数更新：梯度下降优化

Author: Neural Network Math Explorer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simple_latex import display_latex


class SingleNeuron:
    """
    单神经元模型 - 神经网络的最小计算单元
    
    数学模型：
        前向传播: y = activation(w^T · x + b)
        其中：
        - x: 输入向量
        - w: 权重向量
        - b: 偏置
        - activation: 激活函数
    """
    
    def __init__(self, input_size=3, activation='relu', seed=42):
        """
        初始化单神经元
        
        Args:
            input_size: 输入维度
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
            seed: 随机种子
        """
        np.random.seed(seed)
        self.input_size = input_size
        self.activation_name = activation
        
        # 初始化权重和偏置（小随机值）
        self.weights = np.random.randn(input_size) * 0.5
        self.bias = np.random.randn() * 0.1
        
        # 存储计算历史（用于可视化）
        self.forward_history = {}
        self.backward_history = {}
    
    def activation(self, z):
        """激活函数"""
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        else:
            return z
    
    def activation_derivative(self, z):
        """激活函数的导数"""
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = self.activation(z)
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        else:
            return np.ones_like(z)
    
    def forward(self, x):
        """
        前向传播
        
        计算步骤：
        1. 加权和: z = w^T · x + b = sum(w_i * x_i) + b
        2. 激活: y = activation(z)
        
        Args:
            x: 输入向量 (input_size,)
            
        Returns:
            output: 神经元输出
        """
        x = np.array(x, dtype=np.float64)
        
        # 步骤1: 计算加权和
        weighted_sum = np.dot(self.weights, x) + self.bias
        
        # 步骤2: 应用激活函数
        output = self.activation(weighted_sum)
        
        # 保存历史用于可视化和反向传播
        self.forward_history = {
            'input': x.copy(),
            'weights': self.weights.copy(),
            'bias': self.bias,
            'weighted_sum': weighted_sum,
            'activation_derivative': self.activation_derivative(weighted_sum),
            'output': output
        }
        
        return output
    
    def backward(self, upstream_gradient):
        """
        反向传播 - 使用链式法则计算梯度
        
        链式法则：
        ∂L/∂w_i = ∂L/∂y · ∂y/∂z · ∂z/∂w_i
                = upstream_grad · activation'(z) · x_i
        
        ∂L/∂b = ∂L/∂y · ∂y/∂z · ∂z/∂b
              = upstream_grad · activation'(z) · 1
        
        Args:
            upstream_gradient: 来自损失函数的梯度 ∂L/∂y
            
        Returns:
            gradients: 包含所有参数梯度的字典
        """
        # 局部梯度: ∂L/∂z = ∂L/∂y · ∂y/∂z
        local_gradient = upstream_gradient * self.forward_history['activation_derivative']
        
        # 权重梯度: ∂L/∂w = ∂L/∂z · ∂z/∂w = ∂L/∂z · x
        grad_weights = local_gradient * self.forward_history['input']
        
        # 偏置梯度: ∂L/∂b = ∂L/∂z · ∂z/∂b = ∂L/∂z · 1
        grad_bias = local_gradient
        
        # 输入梯度（用于多层网络）: ∂L/∂x = ∂L/∂z · ∂z/∂x = ∂L/∂z · w
        grad_input = local_gradient * self.weights
        
        # 保存梯度历史
        self.backward_history = {
            'upstream_gradient': upstream_gradient,
            'local_gradient': local_gradient,
            'grad_weights': grad_weights,
            'grad_bias': grad_bias,
            'grad_input': grad_input
        }
        
        return {
            'weights': grad_weights,
            'bias': grad_bias,
            'input': grad_input
        }
    
    def update_parameters(self, learning_rate=0.01):
        """
        使用梯度下降更新参数
        
        更新规则：
        w_new = w_old - learning_rate · ∂L/∂w
        b_new = b_old - learning_rate · ∂L/∂b
        
        Args:
            learning_rate: 学习率
        """
        self.weights -= learning_rate * self.backward_history['grad_weights']
        self.bias -= learning_rate * self.backward_history['grad_bias']


def create_computation_table(neuron, precision=4):
    """
    创建前向传播计算步骤表格
    
    Args:
        neuron: SingleNeuron实例
        precision: 数值精度
        
    Returns:
        pd.DataFrame: 计算步骤表格
    """
    history = neuron.forward_history
    steps = []
    
    # 步骤1: 显示输入
    for i, val in enumerate(history['input']):
        steps.append({
            '步骤': f'输入 {i+1}',
            '符号': f'$x_{{{i}}}$',
            '数值': round(val, precision),
            '说明': f'第{i+1}个输入特征'
        })
    
    # 步骤2: 显示权重
    for i, w in enumerate(history['weights']):
        steps.append({
            '步骤': f'权重 {i+1}',
            '符号': f'$w_{{{i}}}$',
            '数值': round(w, precision),
            '说明': f'第{i+1}个权重参数'
        })
    
    # 步骤3: 显示偏置
    steps.append({
        '步骤': '偏置',
        '符号': r'$b$',
        '数值': round(history['bias'], precision),
        '说明': '偏置项'
    })
    
    # 步骤4: 计算加权和
    weighted_parts = [f"({round(w, precision)} × {round(x, precision)})" 
                     for x, w in zip(history['input'], history['weights'])]
    steps.append({
        '步骤': '加权和',
        '符号': r'$z = \sum_{i} w_i \cdot x_i + b$',
        '数值': round(history['weighted_sum'], precision),
        '说明': f'{" + ".join(weighted_parts)} + {round(history["bias"], precision)}'
    })
    
    # 步骤5: 激活函数
    steps.append({
        '步骤': '激活函数',
        '符号': f'$y = {neuron.activation_name}(z)$',
        '数值': round(history['output'], precision),
        '说明': f'{neuron.activation_name}({round(history["weighted_sum"], precision)}) = {round(history["output"], precision)}'
    })
    
    return pd.DataFrame(steps)


def create_gradient_table(neuron, precision=6):
    """
    创建反向传播梯度表格
    
    Args:
        neuron: SingleNeuron实例
        precision: 数值精度
        
    Returns:
        pd.DataFrame: 梯度表格
    """
    backward_hist = neuron.backward_history
    forward_hist = neuron.forward_history
    gradients = []
    
    # 上游梯度
    gradients.append({
        '梯度类型': '上游梯度',
        '符号': r'$\frac{\partial L}{\partial y}$',
        '数值': round(backward_hist['upstream_gradient'], precision),
        '说明': '来自损失函数的梯度（假设值）'
    })
    
    # 激活函数导数
    gradients.append({
        '梯度类型': '激活函数导数',
        '符号': r'$\frac{\partial y}{\partial z}$',
        '数值': round(forward_hist['activation_derivative'], precision),
        '说明': f"{neuron.activation_name}'({round(forward_hist['weighted_sum'], precision)})"
    })
    
    # 局部梯度
    gradients.append({
        '梯度类型': '局部梯度',
        '符号': r'$\frac{\partial L}{\partial z}$',
        '数值': round(backward_hist['local_gradient'], precision),
        '说明': f"= {round(backward_hist['upstream_gradient'], precision)} × {round(forward_hist['activation_derivative'], precision)}"
    })
    
    # 权重梯度
    for i, grad_w in enumerate(backward_hist['grad_weights']):
        gradients.append({
            '梯度类型': f'权重梯度 {i+1}',
            '符号': rf'$\frac{{\partial L}}{{\partial w_{{{i}}}}}$',
            '数值': round(grad_w, precision),
            '说明': f"= {round(backward_hist['local_gradient'], precision)} × {round(forward_hist['input'][i], precision)}"
        })
    
    # 偏置梯度
    gradients.append({
        '梯度类型': '偏置梯度',
        '符号': r'$\frac{\partial L}{\partial b}$',
        '数值': round(backward_hist['grad_bias'], precision),
        '说明': f"= {round(backward_hist['local_gradient'], precision)} × 1"
    })
    
    return pd.DataFrame(gradients)


def visualize_forward_pass(neuron):
    """
    可视化前向传播过程
    
    Args:
        neuron: SingleNeuron实例
        
    Returns:
        plotly Figure对象
    """
    history = neuron.forward_history
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('输入与权重', '加权和计算', '激活函数曲线', '计算流程'),
        specs=[[{'type': 'bar'}, {'type': 'indicator'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 子图1: 输入与权重对比
    x_labels = [f'x[{i}]' for i in range(len(history['input']))]
    fig.add_trace(
        go.Bar(name='输入值', x=x_labels, y=history['input'], 
               marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='权重值', x=x_labels, y=history['weights'], 
               marker_color='lightcoral'),
        row=1, col=1
    )
    
    # 子图2: 加权和指示器
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=history['weighted_sum'],
            title={'text': "加权和 z"},
            delta={'reference': 0},
        ),
        row=1, col=2
    )
    
    # 子图3: 激活函数曲线
    z_range = np.linspace(-3, 3, 100)
    if neuron.activation_name == 'relu':
        y_range = np.maximum(0, z_range)
    elif neuron.activation_name == 'sigmoid':
        y_range = 1 / (1 + np.exp(-z_range))
    elif neuron.activation_name == 'tanh':
        y_range = np.tanh(z_range)
    else:
        y_range = z_range
    
    fig.add_trace(
        go.Scatter(x=z_range, y=y_range, mode='lines', 
                   name=neuron.activation_name, line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[history['weighted_sum']], y=[history['output']], 
                   mode='markers', name='当前点', 
                   marker=dict(size=12, color='red')),
        row=2, col=1
    )
    
    # 子图4: 加权乘积分解
    products = history['input'] * history['weights']
    x_labels_prod = [f'w[{i}]×x[{i}]' for i in range(len(products))]
    fig.add_trace(
        go.Bar(x=x_labels_prod, y=products, 
               marker_color='lightgreen', name='w×x'),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=True, 
                      title_text="单神经元前向传播可视化")
    
    return fig


def visualize_backward_pass(neuron):
    """
    可视化反向传播过程
    
    Args:
        neuron: SingleNeuron实例
        
    Returns:
        plotly Figure对象
    """
    backward_hist = neuron.backward_history
    forward_hist = neuron.forward_history
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('梯度流动', '参数梯度分布'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # 子图1: 梯度流动（从输出到输入）
    gradient_flow = [
        backward_hist['upstream_gradient'],
        backward_hist['local_gradient'],
        np.mean(np.abs(backward_hist['grad_weights']))
    ]
    gradient_labels = ['上游梯度\n∂L/∂y', '局部梯度\n∂L/∂z', '权重梯度\n∂L/∂w']
    
    fig.add_trace(
        go.Bar(x=gradient_labels, y=gradient_flow, 
               marker_color=['red', 'orange', 'yellow']),
        row=1, col=1
    )
    
    # 子图2: 各参数梯度大小
    param_labels = [f'w[{i}]' for i in range(len(backward_hist['grad_weights']))] + ['b']
    param_grads = list(backward_hist['grad_weights']) + [backward_hist['grad_bias']]
    
    fig.add_trace(
        go.Bar(x=param_labels, y=param_grads, 
               marker_color='lightblue'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False,
                      title_text="单神经元反向传播可视化")
    
    return fig


def single_neuron_tab(CHINESE_SUPPORTED=True):
    """
    单神经元可视化标签页 - 主UI界面
    """
    
    if CHINESE_SUPPORTED:
        st.header("🧬 单神经元：理解神经网络的基本单元")
        
        st.markdown("""
        ### 💡 核心思想
        
        神经元是神经网络的**最小计算单元**。理解单个神经元如何工作，就能理解整个神经网络的运作原理。
        
        **一个神经元做什么？**
        1. 接收多个输入信号
        2. 对每个输入加权求和（加上偏置）
        3. 通过激活函数引入非线性
        4. 输出处理后的信号
        """)
        
        # 显示数学公式
        st.markdown("### 📐 数学表达")
        st.markdown(r"""

        $\text{前向传播：}$
        $$
        \begin{aligned}
        z &= \sum_{i=0}^{n} w_i \cdot x_i + b = w^T x + b \\
        y &= \text{activation}(z)
        \end{aligned}
        $$
        
        $\text{反向传播（链式法则）：}$

        $$
        \begin{aligned}
        \frac{\partial L}{\partial w_i} &= \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_i} 
        = \frac{\partial L}{\partial y} \cdot \text{activation}'(z) \cdot x_i \\
        \frac{\partial L}{\partial b} &= \frac{\partial L}{\partial y} \cdot \text{activation}'(z)
        \end{aligned}
        $$
        """)
        
        st.markdown("---")
        
        # 配置区域
        st.subheader("⚙️ 配置神经元")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_size = st.slider("输入维度", 1, 5, 3, 
                                   help="神经元接收多少个输入")
            activation = st.selectbox(
                "激活函数",
                ['relu', 'sigmoid', 'tanh'],
                help="选择激活函数类型"
            )
        
        with col2:
            seed = st.number_input("随机种子", 0, 100, 42, 
                                   help="用于初始化权重")
            learning_rate = st.slider("学习率", 0.001, 0.5, 0.01, 0.001,
                                      help="梯度下降的步长")
        
        # 创建神经元
        neuron = SingleNeuron(input_size=input_size, activation=activation, seed=seed)
        
        st.markdown("---")
        
        # 输入数据
        st.subheader("📥 输入数据")
        
        st.write("设置输入值：")
        input_cols = st.columns(input_size)
        input_data = []
        for i, col in enumerate(input_cols):
            with col:
                val = st.number_input(
                    f"$x_{{{i}}}$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"input_{i}"
                )
                input_data.append(val)
        
        input_data = np.array(input_data)
        
        # 显示当前参数
        st.subheader("🎯 当前参数")
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.write("**权重向量 w:**")
            weights_df = pd.DataFrame({
                '索引': [f'w[{i}]' for i in range(input_size)],
                '数值': [f'{w:.6f}' for w in neuron.weights]
            })
            st.markdown(weights_df.to_markdown(index=False))
        
        with param_col2:
            st.write("**偏置 b:**")
            st.metric("bias", f"{neuron.bias:.6f}")
        
        st.markdown("---")
        
        # 上游梯度设置
        st.subheader("🎯 上游梯度设置")
        upstream_grad = st.number_input(
            "上游梯度 (∂L/∂y)",
            value=1.0,
            format="%.6f",
            help="假设这是从损失函数传回的梯度。在真实训练中，这来自损失函数对输出的导数。"
        )
        
        st.markdown("---")
        
        # 执行完整的前向-反向-更新流程
        if st.button("🚀 执行完整计算流程（前向→反向→更新）", type="primary"):
            # ==================== 前向传播 ====================
            st.subheader("➡️ 1. 前向传播")
            output = neuron.forward(input_data)
            
            st.success(f"✅ 神经元输出: **{output:.6f}**")
            
            # 计算步骤表格
            with st.expander("📋 详细计算步骤", expanded=True):
                comp_table = create_computation_table(neuron)
                st.markdown(comp_table.to_markdown(index=False))
            
            # 可视化
            with st.expander("📊 前向传播可视化", expanded=True):
                fig_forward = visualize_forward_pass(neuron)
                st.plotly_chart(fig_forward, use_container_width=True)
            
            # ==================== 反向传播 ====================
            st.markdown("---")
            st.subheader("⬅️ 2. 反向传播")
            
            # 保存旧参数用于对比
            old_weights = neuron.weights.copy()
            old_bias = neuron.bias
            
            gradients = neuron.backward(upstream_grad)
            
            st.success("✅ 梯度计算完成")
            
            # 梯度表格
            with st.expander("📋 梯度详细信息", expanded=True):
                grad_table = create_gradient_table(neuron)
                st.markdown(grad_table.to_markdown(index=False))
            
            # 可视化
            with st.expander("📊 梯度流动可视化", expanded=True):
                fig_backward = visualize_backward_pass(neuron)
                st.plotly_chart(fig_backward, use_container_width=True)
            
            # ==================== 参数更新 ====================
            st.markdown("---")
            st.subheader("📊 3. 参数更新（梯度下降）")
            
            # 更新参数
            neuron.update_parameters(learning_rate)
            
            st.success("✅ 参数已更新")
            
            # 显示更新详情
            st.write("**参数变化对比：**")
            
            update_data = []
            for i in range(input_size):
                delta = neuron.weights[i] - old_weights[i]
                update_data.append({
                    '参数': f'$w_{{{i}}}$',
                    '更新前': f'{old_weights[i]:.6f}',
                    '梯度': f'{gradients["weights"][i]:.6f}',
                    '更新量': f'{-learning_rate * gradients["weights"][i]:.6f}',
                    '更新后': f'{neuron.weights[i]:.6f}',
                    '变化': f'{delta:.6f}'
                })
            
            delta_bias = neuron.bias - old_bias
            update_data.append({
                '参数': '$b$',
                '更新前': f'{old_bias:.6f}',
                '梯度': f'{gradients["bias"]:.6f}',
                '更新量': f'{-learning_rate * gradients["bias"]:.6f}',
                '更新后': f'{neuron.bias:.6f}',
                '变化': f'{delta_bias:.6f}'
            })
            
            update_df = pd.DataFrame(update_data)
            st.markdown(update_df.to_markdown(index=False))
            
            st.info(f"💡 **更新规则**: $\\theta_{{\text{{new}}}} = \\theta_{{\text{{old}}}} - \\alpha \\cdot \\frac{{\\partial L}}{{\\partial \\theta}}$，其中学习率 $\\alpha = {learning_rate}$")
            
            # ==================== 总结 ====================
            st.markdown("---")
            st.subheader("📈 完整流程总结")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "前向传播输出",
                    f"{output:.6f}",
                    help="神经元的最终输出"
                )
            
            with col2:
                avg_grad = np.mean(np.abs(gradients['weights']))
                st.metric(
                    "平均权重梯度",
                    f"{avg_grad:.6f}",
                    help="权重梯度的平均绝对值"
                )
            
            with col3:
                total_change = np.linalg.norm(neuron.weights - old_weights)
                st.metric(
                    "参数变化量",
                    f"{total_change:.6f}",
                    help="所有权重变化的L2范数"
                )
            
            st.success("""
            ✅ **完整训练步骤已完成！**
            
            这就是神经网络训练的一个完整迭代：
            1. **前向传播**：输入 → 加权和 → 激活 → 输出
            2. **反向传播**：计算每个参数的梯度（链式法则）
            3. **参数更新**：沿着梯度的反方向更新参数
            
            在实际训练中，这个过程会重复成千上万次，每次使用不同的训练样本。
            """)
        
        # 教学说明
        st.markdown("---")
        with st.expander("📚 详细说明：神经元如何工作", expanded=False):
            st.markdown("""
            ### 🔍 深入理解
            
            #### 1. 前向传播（Forward Propagation）
            
            神经元接收输入后，执行两步计算：
            
            **步骤1: 线性组合（加权和）**
            - 每个输入 $x_i$ 乘以对应的权重 $w_i$
            - 将所有乘积相加，再加上偏置 $b$
            - 结果：$z = w_0 x_0 + w_1 x_1 + ... + w_n x_n + b$
            
            **步骤2: 非线性激活**
            - 将 $z$ 通过激活函数转换
            - 激活函数引入非线性，让网络能学习复杂模式
            - 常用激活函数：
                - **ReLU**: $f(z) = \\max(0, z)$ - 简单有效，最常用
                - **Sigmoid**: $f(z) = \\frac{1}{1+e^{-z}}$ - 输出0到1，用于概率
                - **Tanh**: $f(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$ - 输出-1到1，零中心化
            
            #### 2. 反向传播（Backpropagation）
            
            使用**链式法则**计算每个参数的梯度：
            
            **核心思想**: 从输出往回传播误差
            
            - **上游梯度** $\\frac{\\partial L}{\\partial y}$: 来自损失函数或下一层
            - **激活函数导数** $\\frac{\\partial y}{\\partial z}$: 激活函数在当前点的斜率
            - **局部梯度** $\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial z}$
            - **参数梯度**: 
                - $\\frac{\\partial L}{\\partial w_i} = \\frac{\\partial L}{\\partial z} \\cdot x_i$
                - $\\frac{\\partial L}{\\partial b} = \\frac{\\partial L}{\\partial z}$
            
            #### 3. 参数更新（Parameter Update）
            
            使用梯度下降优化参数：
            
            $$
            w_{\\text{new}} = w_{\\text{old}} - \\alpha \\cdot \\frac{\\partial L}{\\partial w}
            $$
            
            - $\\alpha$ 是学习率，控制更新步长
            - 梯度指向误差增加的方向，所以要减去梯度
            - 迭代多次后，参数会逐渐优化
            
            #### 4. 关键洞察
            
            - **权重**决定每个输入的重要性
            - **偏置**调整神经元的激活阈值
            - **激活函数**引入非线性，是深度学习的关键
            - **梯度**指示参数应该如何调整
            - **学习率**控制学习速度（过大震荡，过小缓慢）
            
            #### 5. 从单神经元到神经网络
            
            - 多个神经元并行 → **层（Layer）**
            - 多个层堆叠 → **深度神经网络（DNN）**
            - 每一层做类似的计算：$\\text{output} = \\text{activation}(W \\cdot \\text{input} + b)$
            - 通过反向传播，所有层的参数同时更新
            
            **理解单个神经元 = 理解整个神经网络！**
            """)
    
    else:
        st.header("🧬 Single Neuron: Understanding the Basic Unit")
        st.info("English version - Coming soon!")


if __name__ == "__main__":
    # 测试代码
    neuron = SingleNeuron(input_size=3, activation='relu')
    x = np.array([0.5, -0.3, 0.2])
    
    # 前向传播
    output = neuron.forward(x)
    print(f"Forward pass output: {output:.6f}")
    
    # 反向传播
    gradients = neuron.backward(upstream_gradient=1.0)
    print(f"Gradients: {gradients}")
    
    # 参数更新
    neuron.update_parameters(learning_rate=0.01)
    print("Parameters updated!")

