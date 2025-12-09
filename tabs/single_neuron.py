"""
单神经元可视化模块 - Single Neuron Visualization
==================================================

通过单个神经元的视角，深入理解神经网络的工作原理。

核心概念：
1. 神经元是神经网络的基本计算单元
2. 前向传播：加权和 + 激活函数
3. 反向传播：链式法则计算梯度
4. 参数更新：梯度下降优化

支持多种神经元类型：
- 全连接神经元（Dense/FC）
- 卷积神经元（Conv）
- 循环神经元（RNN）
- GRU神经元
- LSTM神经元
- 注意力机制（Attention）

v2.2.0 新增：
- 数值稳定性自动检测
- 梯度验证
- 问题诊断和解决方案

Author: Neural Network Math Explorer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simple_latex import display_latex
from utils.numerical_stability_checker import (
    StabilityChecker,
    compute_numerical_gradient,
)


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

    def __init__(self, input_size=3, activation="relu", seed=42):
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
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(z)
        else:
            return z

    def activation_derivative(self, z):
        """激活函数的导数"""
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = self.activation(z)
            return s * (1 - s)
        elif self.activation_name == "tanh":
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
            "input": x.copy(),
            "weights": self.weights.copy(),
            "bias": self.bias,
            "weighted_sum": weighted_sum,
            "activation_derivative": self.activation_derivative(weighted_sum),
            "output": output,
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
        local_gradient = (
            upstream_gradient * self.forward_history["activation_derivative"]
        )

        # 权重梯度: ∂L/∂w = ∂L/∂z · ∂z/∂w = ∂L/∂z · x
        grad_weights = local_gradient * self.forward_history["input"]

        # 偏置梯度: ∂L/∂b = ∂L/∂z · ∂z/∂b = ∂L/∂z · 1
        grad_bias = local_gradient

        # 输入梯度（用于多层网络）: ∂L/∂x = ∂L/∂z · ∂z/∂x = ∂L/∂z · w
        grad_input = local_gradient * self.weights

        # 保存梯度历史
        self.backward_history = {
            "upstream_gradient": upstream_gradient,
            "local_gradient": local_gradient,
            "grad_weights": grad_weights,
            "grad_bias": grad_bias,
            "grad_input": grad_input,
        }

        return {"weights": grad_weights, "bias": grad_bias, "input": grad_input}

    def update_parameters(self, learning_rate=0.01):
        """
        使用梯度下降更新参数

        更新规则：
        w_new = w_old - learning_rate · ∂L/∂w
        b_new = b_old - learning_rate · ∂L/∂b

        Args:
            learning_rate: 学习率
        """
        self.weights -= learning_rate * self.backward_history["grad_weights"]
        self.bias -= learning_rate * self.backward_history["grad_bias"]


class ConvNeuron:
    """
    卷积神经元 - 用于图像处理

    数学模型：
        前向传播: y = activation(sum(kernel * patch) + b)
        其中：
        - patch: 输入图像的局部区域 (kernel_size × kernel_size)
        - kernel: 卷积核 (权重)
        - b: 偏置
    """

    def __init__(self, kernel_size=3, activation="relu", seed=42):
        np.random.seed(seed)
        self.kernel_size = kernel_size
        self.activation_name = activation

        # 初始化卷积核和偏置
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.5
        self.bias = np.random.randn() * 0.1

        self.forward_history = {}
        self.backward_history = {}

    def activation(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(z)
        return z

    def activation_derivative(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = self.activation(z)
            return s * (1 - s)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(z) ** 2
        return np.ones_like(z)

    def forward(self, x):
        """
        前向传播 - 卷积操作

        Args:
            x: 输入图像块 (kernel_size, kernel_size)
        """
        x = np.array(x, dtype=np.float64)

        # 卷积操作：逐元素相乘再求和
        weighted_sum = np.sum(self.kernel * x) + self.bias
        output = self.activation(weighted_sum)

        self.forward_history = {
            "input": x.copy(),
            "kernel": self.kernel.copy(),
            "bias": self.bias,
            "weighted_sum": weighted_sum,
            "activation_derivative": self.activation_derivative(weighted_sum),
            "output": output,
        }

        return output

    def backward(self, upstream_gradient):
        """反向传播"""
        local_gradient = (
            upstream_gradient * self.forward_history["activation_derivative"]
        )

        # 卷积核梯度
        grad_kernel = local_gradient * self.forward_history["input"]
        grad_bias = local_gradient
        grad_input = local_gradient * self.kernel

        self.backward_history = {
            "upstream_gradient": upstream_gradient,
            "local_gradient": local_gradient,
            "grad_kernel": grad_kernel,
            "grad_bias": grad_bias,
            "grad_input": grad_input,
        }

        return {"kernel": grad_kernel, "bias": grad_bias, "input": grad_input}

    def update_parameters(self, learning_rate=0.01):
        """更新参数"""
        self.kernel -= learning_rate * self.backward_history["grad_kernel"]
        self.bias -= learning_rate * self.backward_history["grad_bias"]


class RNNNeuron:
    """
    循环神经元 - 用于序列处理

    数学模型：
        h_t = activation(W_hh * h_{t-1} + W_xh * x_t + b)
        其中：
        - h_t: 当前隐藏状态
        - h_{t-1}: 前一时刻隐藏状态
        - x_t: 当前输入
    """

    def __init__(self, input_size=3, hidden_size=3, activation="tanh", seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation

        # 初始化权重
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.5  # 输入到隐藏
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.5  # 隐藏到隐藏
        self.b_h = np.random.randn(hidden_size) * 0.1

        # 初始化隐藏状态
        self.h = np.zeros(hidden_size)

        self.forward_history = {}
        self.backward_history = {}

    def activation(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(z)
        return z

    def activation_derivative(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = self.activation(z)
            return s * (1 - s)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(z) ** 2
        return np.ones_like(z)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 当前时刻输入 (input_size,)
        """
        x = np.array(x, dtype=np.float64)
        h_prev = self.h.copy()

        # RNN 计算
        weighted_sum = np.dot(self.W_hh, h_prev) + np.dot(self.W_xh, x) + self.b_h
        self.h = self.activation(weighted_sum)

        self.forward_history = {
            "input": x.copy(),
            "h_prev": h_prev,
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "weighted_sum": weighted_sum,
            "activation_derivative": self.activation_derivative(weighted_sum),
            "output": self.h.copy(),
        }

        return self.h

    def backward(self, upstream_gradient):
        """反向传播"""
        local_gradient = (
            upstream_gradient * self.forward_history["activation_derivative"]
        )

        # 计算各参数梯度
        grad_W_xh = np.outer(local_gradient, self.forward_history["input"])
        grad_W_hh = np.outer(local_gradient, self.forward_history["h_prev"])
        grad_b_h = local_gradient

        # 传递给前一时刻的梯度
        grad_h_prev = np.dot(self.W_hh.T, local_gradient)
        grad_input = np.dot(self.W_xh.T, local_gradient)

        self.backward_history = {
            "upstream_gradient": upstream_gradient,
            "local_gradient": local_gradient,
            "grad_W_xh": grad_W_xh,
            "grad_W_hh": grad_W_hh,
            "grad_b_h": grad_b_h,
            "grad_h_prev": grad_h_prev,
            "grad_input": grad_input,
        }

        return {
            "W_xh": grad_W_xh,
            "W_hh": grad_W_hh,
            "b_h": grad_b_h,
            "h_prev": grad_h_prev,
            "input": grad_input,
        }

    def update_parameters(self, learning_rate=0.01):
        """更新参数"""
        self.W_xh -= learning_rate * self.backward_history["grad_W_xh"]
        self.W_hh -= learning_rate * self.backward_history["grad_W_hh"]
        self.b_h -= learning_rate * self.backward_history["grad_b_h"]

    def reset_hidden(self):
        """重置隐藏状态"""
        self.h = np.zeros(self.hidden_size)


class GRUNeuron:
    """
    GRU神经元 - 门控循环单元（LSTM的简化版）

    数学模型：
        重置门: r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        更新门: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        候选隐藏状态: h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
        隐藏状态: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    GRU vs LSTM:
    - GRU只有2个门（重置、更新）vs LSTM的3个门（遗忘、输入、输出）
    - GRU没有单独的细胞状态，直接更新隐藏状态
    - GRU参数量更少，训练更快
    - 性能通常与LSTM相当
    """

    def __init__(self, input_size=3, hidden_size=3, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined_size = hidden_size + input_size

        # 初始化3组权重（重置门、更新门、候选隐藏状态）
        self.W_r = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_r = np.random.randn(hidden_size) * 0.1

        self.W_z = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_z = np.random.randn(hidden_size) * 0.1

        self.W_h = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_h = np.random.randn(hidden_size) * 0.1

        # 初始化隐藏状态
        self.h = np.zeros(hidden_size)

        self.forward_history = {}
        self.backward_history = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def forward(self, x):
        """
        GRU前向传播

        Args:
            x: 当前时刻输入 (input_size,)
        """
        x = np.array(x, dtype=np.float64)
        h_prev = self.h.copy()

        # 拼接输入
        combined = np.concatenate([h_prev, x])

        # 重置门
        r_t = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)

        # 更新门
        z_t = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)

        # 候选隐藏状态（使用重置门）
        combined_reset = np.concatenate([r_t * h_prev, x])
        h_tilde = np.tanh(np.dot(self.W_h, combined_reset) + self.b_h)

        # 更新隐藏状态（使用更新门）
        self.h = (1 - z_t) * h_prev + z_t * h_tilde

        # 保存前向传播历史
        self.forward_history = {
            "input": x.copy(),
            "h_prev": h_prev,
            "combined": combined,
            "r_t": r_t,
            "z_t": z_t,
            "combined_reset": combined_reset,
            "h_tilde": h_tilde,
            "output": self.h.copy(),
        }

        return self.h

    def backward(self, upstream_gradient):
        """GRU反向传播（简化版）"""
        # 这是一个简化的反向传播
        dh = upstream_gradient
        h_prev = self.forward_history["h_prev"]
        h_tilde = self.forward_history["h_tilde"]
        z_t = self.forward_history["z_t"]
        r_t = self.forward_history["r_t"]

        # 更新门梯度
        dz_t = dh * (h_tilde - h_prev)

        # 候选隐藏状态梯度
        dh_tilde = dh * z_t

        # 重置门梯度（简化）
        dr_t = (
            np.dot(
                self.W_h[:, : self.hidden_size].T,
                dh_tilde
                * self.tanh_derivative(
                    np.dot(self.W_h, self.forward_history["combined_reset"]) + self.b_h
                ),
            )
            * h_prev
        )

        # 权重梯度
        grad_W_r = np.outer(
            dr_t
            * self.sigmoid_derivative(
                np.dot(self.W_r, self.forward_history["combined"]) + self.b_r
            ),
            self.forward_history["combined"],
        )
        grad_b_r = dr_t * self.sigmoid_derivative(
            np.dot(self.W_r, self.forward_history["combined"]) + self.b_r
        )

        grad_W_z = np.outer(
            dz_t
            * self.sigmoid_derivative(
                np.dot(self.W_z, self.forward_history["combined"]) + self.b_z
            ),
            self.forward_history["combined"],
        )
        grad_b_z = dz_t * self.sigmoid_derivative(
            np.dot(self.W_z, self.forward_history["combined"]) + self.b_z
        )

        grad_W_h = np.outer(
            dh_tilde
            * self.tanh_derivative(
                np.dot(self.W_h, self.forward_history["combined_reset"]) + self.b_h
            ),
            self.forward_history["combined_reset"],
        )
        grad_b_h = dh_tilde * self.tanh_derivative(
            np.dot(self.W_h, self.forward_history["combined_reset"]) + self.b_h
        )

        self.backward_history = {
            "grad_W_r": grad_W_r,
            "grad_b_r": grad_b_r,
            "grad_W_z": grad_W_z,
            "grad_b_z": grad_b_z,
            "grad_W_h": grad_W_h,
            "grad_b_h": grad_b_h,
        }

        return self.backward_history

    def update_parameters(self, learning_rate=0.01):
        """更新所有参数"""
        self.W_r -= learning_rate * self.backward_history["grad_W_r"]
        self.b_r -= learning_rate * self.backward_history["grad_b_r"]
        self.W_z -= learning_rate * self.backward_history["grad_W_z"]
        self.b_z -= learning_rate * self.backward_history["grad_b_z"]
        self.W_h -= learning_rate * self.backward_history["grad_W_h"]
        self.b_h -= learning_rate * self.backward_history["grad_b_h"]

    def reset_state(self):
        """重置隐藏状态"""
        self.h = np.zeros(self.hidden_size)


class LSTMNeuron:
    """
    LSTM神经元 - 长短期记忆网络

    数学模型：
        遗忘门: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        输入门: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        候选值: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
        细胞状态: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        输出门: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        隐藏状态: h_t = o_t ⊙ tanh(C_t)
    """

    def __init__(self, input_size=3, hidden_size=3, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined_size = hidden_size + input_size

        # 初始化4组权重（遗忘门、输入门、候选值、输出门）
        self.W_f = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_f = np.random.randn(hidden_size) * 0.1

        self.W_i = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_i = np.random.randn(hidden_size) * 0.1

        self.W_C = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_C = np.random.randn(hidden_size) * 0.1

        self.W_o = np.random.randn(hidden_size, combined_size) * 0.5
        self.b_o = np.random.randn(hidden_size) * 0.1

        # 初始化状态
        self.h = np.zeros(hidden_size)
        self.C = np.zeros(hidden_size)

        self.forward_history = {}
        self.backward_history = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def forward(self, x):
        """
        LSTM前向传播

        Args:
            x: 当前时刻输入 (input_size,)
        """
        x = np.array(x, dtype=np.float64)
        h_prev = self.h.copy()
        C_prev = self.C.copy()

        # 拼接输入
        combined = np.concatenate([h_prev, x])

        # 遗忘门
        f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)

        # 输入门
        i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)

        # 候选值
        C_tilde = np.tanh(np.dot(self.W_C, combined) + self.b_C)

        # 更新细胞状态
        self.C = f_t * C_prev + i_t * C_tilde

        # 输出门
        o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)

        # 更新隐藏状态
        self.h = o_t * np.tanh(self.C)

        # 保存前向传播历史
        self.forward_history = {
            "input": x.copy(),
            "h_prev": h_prev,
            "C_prev": C_prev,
            "combined": combined,
            "f_t": f_t,
            "i_t": i_t,
            "C_tilde": C_tilde,
            "C_t": self.C.copy(),
            "o_t": o_t,
            "output": self.h.copy(),
        }

        return self.h

    def backward(self, upstream_gradient):
        """LSTM反向传播（简化版）"""
        # 这是一个简化的反向传播，实际LSTM的完整BPTT更复杂
        dh = upstream_gradient

        # 输出门梯度
        dC = (
            dh
            * self.forward_history["o_t"]
            * self.tanh_derivative(self.forward_history["C_t"])
        )
        do_t = dh * np.tanh(self.forward_history["C_t"])

        # 候选值梯度
        dC_tilde = dC * self.forward_history["i_t"]
        di_t = dC * self.forward_history["C_tilde"]
        df_t = dC * self.forward_history["C_prev"]

        # 权重梯度
        combined = self.forward_history["combined"]

        grad_W_f = np.outer(
            df_t * self.sigmoid_derivative(np.dot(self.W_f, combined) + self.b_f),
            combined,
        )
        grad_b_f = df_t * self.sigmoid_derivative(np.dot(self.W_f, combined) + self.b_f)

        grad_W_i = np.outer(
            di_t * self.sigmoid_derivative(np.dot(self.W_i, combined) + self.b_i),
            combined,
        )
        grad_b_i = di_t * self.sigmoid_derivative(np.dot(self.W_i, combined) + self.b_i)

        grad_W_C = np.outer(
            dC_tilde * self.tanh_derivative(np.dot(self.W_C, combined) + self.b_C),
            combined,
        )
        grad_b_C = dC_tilde * self.tanh_derivative(
            np.dot(self.W_C, combined) + self.b_C
        )

        grad_W_o = np.outer(
            do_t * self.sigmoid_derivative(np.dot(self.W_o, combined) + self.b_o),
            combined,
        )
        grad_b_o = do_t * self.sigmoid_derivative(np.dot(self.W_o, combined) + self.b_o)

        self.backward_history = {
            "grad_W_f": grad_W_f,
            "grad_b_f": grad_b_f,
            "grad_W_i": grad_W_i,
            "grad_b_i": grad_b_i,
            "grad_W_C": grad_W_C,
            "grad_b_C": grad_b_C,
            "grad_W_o": grad_W_o,
            "grad_b_o": grad_b_o,
        }

        return self.backward_history

    def update_parameters(self, learning_rate=0.01):
        """更新所有参数"""
        self.W_f -= learning_rate * self.backward_history["grad_W_f"]
        self.b_f -= learning_rate * self.backward_history["grad_b_f"]
        self.W_i -= learning_rate * self.backward_history["grad_W_i"]
        self.b_i -= learning_rate * self.backward_history["grad_b_i"]
        self.W_C -= learning_rate * self.backward_history["grad_W_C"]
        self.b_C -= learning_rate * self.backward_history["grad_b_C"]
        self.W_o -= learning_rate * self.backward_history["grad_W_o"]
        self.b_o -= learning_rate * self.backward_history["grad_b_o"]

    def reset_state(self):
        """重置隐藏状态和细胞状态"""
        self.h = np.zeros(self.hidden_size)
        self.C = np.zeros(self.hidden_size)


class AttentionNeuron:
    """
    注意力机制神经元 - 简化的单头注意力

    数学模型：
        Query: Q = x · W_Q
        Key: K = x · W_K
        Value: V = x · W_V
        Attention Score: score = Q · K^T / sqrt(d_k)
        Attention Weight: α = softmax(score)
        Output: y = α · V
    """

    def __init__(self, input_size=3, d_model=3, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.d_model = d_model

        # Q, K, V 投影矩阵
        self.W_Q = np.random.randn(d_model, input_size) * 0.5
        self.W_K = np.random.randn(d_model, input_size) * 0.5
        self.W_V = np.random.randn(d_model, input_size) * 0.5

        self.forward_history = {}
        self.backward_history = {}

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, query, keys, values):
        """
        注意力前向传播

        Args:
            query: 查询向量 (input_size,)
            keys: 键向量列表，每个 (input_size,)
            values: 值向量列表，每个 (input_size,)
        """
        query = np.array(query, dtype=np.float64)
        keys = np.array(keys, dtype=np.float64)
        values = np.array(values, dtype=np.float64)

        # 投影到 Q, K, V
        Q = np.dot(self.W_Q, query)
        K = np.array([np.dot(self.W_K, k) for k in keys])
        V = np.array([np.dot(self.W_V, v) for v in values])

        # 计算注意力分数
        scores = np.dot(K, Q) / np.sqrt(self.d_model)

        # Softmax得到注意力权重
        attention_weights = self.softmax(scores)

        # 加权求和
        output = np.dot(attention_weights, V)

        self.forward_history = {
            "query": query.copy(),
            "keys": keys.copy(),
            "values": values.copy(),
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "attention_weights": attention_weights,
            "output": output,
        }

        return output, attention_weights

    def backward(self, upstream_gradient):
        """注意力反向传播（简化版）"""
        # 简化的梯度计算
        grad_V = np.outer(self.forward_history["attention_weights"], upstream_gradient)
        grad_W_V = np.zeros_like(self.W_V)

        for i, v in enumerate(self.forward_history["values"]):
            grad_W_V += np.outer(grad_V[i], v)

        # 其他梯度计算（简化）
        grad_W_Q = np.outer(upstream_gradient, self.forward_history["query"])
        grad_W_K = np.outer(upstream_gradient, self.forward_history["query"])

        self.backward_history = {
            "grad_W_Q": grad_W_Q,
            "grad_W_K": grad_W_K,
            "grad_W_V": grad_W_V,
        }

        return self.backward_history

    def update_parameters(self, learning_rate=0.01):
        """更新参数"""
        self.W_Q -= learning_rate * self.backward_history["grad_W_Q"]
        self.W_K -= learning_rate * self.backward_history["grad_W_K"]
        self.W_V -= learning_rate * self.backward_history["grad_W_V"]


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
    for i, val in enumerate(history["input"]):
        steps.append(
            {
                "步骤": f"输入 {i+1}",
                "符号": f"$x_{{{i}}}$",
                "数值": round(val, precision),
                "说明": f"第{i+1}个输入特征",
            }
        )

    # 步骤2: 显示权重
    for i, w in enumerate(history["weights"]):
        steps.append(
            {
                "步骤": f"权重 {i+1}",
                "符号": f"$w_{{{i}}}$",
                "数值": round(w, precision),
                "说明": f"第{i+1}个权重参数",
            }
        )

    # 步骤3: 显示偏置
    steps.append(
        {
            "步骤": "偏置",
            "符号": r"$b$",
            "数值": round(history["bias"], precision),
            "说明": "偏置项",
        }
    )

    # 步骤4: 计算加权和
    weighted_parts = [
        f"({round(w, precision)} × {round(x, precision)})"
        for x, w in zip(history["input"], history["weights"])
    ]
    steps.append(
        {
            "步骤": "加权和",
            "符号": r"$z = \sum_{i} w_i \cdot x_i + b$",
            "数值": round(history["weighted_sum"], precision),
            "说明": f'{" + ".join(weighted_parts)} + {round(history["bias"], precision)}',
        }
    )

    # 步骤5: 激活函数
    steps.append(
        {
            "步骤": "激活函数",
            "符号": f"$y = {neuron.activation_name}(z)$",
            "数值": round(history["output"], precision),
            "说明": f'{neuron.activation_name}({round(history["weighted_sum"], precision)}) = {round(history["output"], precision)}',
        }
    )

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
    gradients.append(
        {
            "梯度类型": "上游梯度",
            "符号": r"$\frac{\partial L}{\partial y}$",
            "数值": round(backward_hist["upstream_gradient"], precision),
            "说明": "来自损失函数的梯度（假设值）",
        }
    )

    # 激活函数导数
    gradients.append(
        {
            "梯度类型": "激活函数导数",
            "符号": r"$\frac{\partial y}{\partial z}$",
            "数值": round(forward_hist["activation_derivative"], precision),
            "说明": f"{neuron.activation_name}'({round(forward_hist['weighted_sum'], precision)})",
        }
    )

    # 局部梯度
    gradients.append(
        {
            "梯度类型": "局部梯度",
            "符号": r"$\frac{\partial L}{\partial z}$",
            "数值": round(backward_hist["local_gradient"], precision),
            "说明": f"= {round(backward_hist['upstream_gradient'], precision)} × {round(forward_hist['activation_derivative'], precision)}",
        }
    )

    # 权重梯度
    for i, grad_w in enumerate(backward_hist["grad_weights"]):
        gradients.append(
            {
                "梯度类型": f"权重梯度 {i+1}",
                "符号": rf"$\frac{{\partial L}}{{\partial w_{{{i}}}}}$",
                "数值": round(grad_w, precision),
                "说明": f"= {round(backward_hist['local_gradient'], precision)} × {round(forward_hist['input'][i], precision)}",
            }
        )

    # 偏置梯度
    gradients.append(
        {
            "梯度类型": "偏置梯度",
            "符号": r"$\frac{\partial L}{\partial b}$",
            "数值": round(backward_hist["grad_bias"], precision),
            "说明": f"= {round(backward_hist['local_gradient'], precision)} × 1",
        }
    )

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
        rows=2,
        cols=2,
        subplot_titles=("输入与权重", "加权和计算", "激活函数曲线", "计算流程"),
        specs=[
            [{"type": "bar"}, {"type": "indicator"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # 子图1: 输入与权重对比
    x_labels = [f"x[{i}]" for i in range(len(history["input"]))]
    fig.add_trace(
        go.Bar(name="输入值", x=x_labels, y=history["input"], marker_color="lightblue"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="权重值", x=x_labels, y=history["weights"], marker_color="lightcoral"
        ),
        row=1,
        col=1,
    )

    # 子图2: 加权和指示器
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=history["weighted_sum"],
            title={"text": "加权和 z"},
            delta={"reference": 0},
        ),
        row=1,
        col=2,
    )

    # 子图3: 激活函数曲线
    z_range = np.linspace(-3, 3, 100)
    if neuron.activation_name == "relu":
        y_range = np.maximum(0, z_range)
    elif neuron.activation_name == "sigmoid":
        y_range = 1 / (1 + np.exp(-z_range))
    elif neuron.activation_name == "tanh":
        y_range = np.tanh(z_range)
    else:
        y_range = z_range

    fig.add_trace(
        go.Scatter(
            x=z_range,
            y=y_range,
            mode="lines",
            name=neuron.activation_name,
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[history["weighted_sum"]],
            y=[history["output"]],
            mode="markers",
            name="当前点",
            marker=dict(size=12, color="red"),
        ),
        row=2,
        col=1,
    )

    # 子图4: 加权乘积分解
    products = history["input"] * history["weights"]
    x_labels_prod = [f"w[{i}]×x[{i}]" for i in range(len(products))]
    fig.add_trace(
        go.Bar(x=x_labels_prod, y=products, marker_color="lightgreen", name="w×x"),
        row=2,
        col=2,
    )

    fig.update_layout(height=700, showlegend=True, title_text="单神经元前向传播可视化")

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
        rows=1,
        cols=2,
        subplot_titles=("梯度流动", "参数梯度分布"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # 子图1: 梯度流动（从输出到输入）
    gradient_flow = [
        backward_hist["upstream_gradient"],
        backward_hist["local_gradient"],
        np.mean(np.abs(backward_hist["grad_weights"])),
    ]
    gradient_labels = ["上游梯度\n∂L/∂y", "局部梯度\n∂L/∂z", "权重梯度\n∂L/∂w"]

    fig.add_trace(
        go.Bar(
            x=gradient_labels, y=gradient_flow, marker_color=["red", "orange", "yellow"]
        ),
        row=1,
        col=1,
    )

    # 子图2: 各参数梯度大小
    param_labels = [f"w[{i}]" for i in range(len(backward_hist["grad_weights"]))] + [
        "b"
    ]
    param_grads = list(backward_hist["grad_weights"]) + [backward_hist["grad_bias"]]

    fig.add_trace(
        go.Bar(x=param_labels, y=param_grads, marker_color="lightblue"), row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False, title_text="单神经元反向传播可视化")

    return fig


def render_dense_neuron():
    """渲染全连接神经元演示"""
    st.subheader("📐 全连接神经元（Dense/Fully Connected）")

    st.markdown(
        """
    **工作原理**：
    1. 接收多个输入信号
    2. 对每个输入加权求和（加上偏置）
    3. 通过激活函数引入非线性
    4. 输出处理后的信号
    
    **数学表达**：
    $$
    \\begin{aligned}
    z &= \\sum_{i=0}^{n} w_i \\cdot x_i + b = w^T x + b \\\\
    y &= \\text{activation}(z)
    \\end{aligned}
    $$
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置神经元")

    col1, col2 = st.columns(2)

    with col1:
        input_size = st.slider("输入维度", 1, 5, 3, help="神经元接收多少个输入")
        activation = st.selectbox(
            "激活函数", ["relu", "sigmoid", "tanh"], help="选择激活函数类型"
        )

    with col2:
        seed = st.number_input("随机种子", 0, 100, 42, help="用于初始化权重")
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长"
        )

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
                key=f"dense_input_{i}",
            )
            input_data.append(val)

    input_data = np.array(input_data)

    # 显示当前参数
    st.subheader("🎯 当前参数")
    param_col1, param_col2 = st.columns(2)

    with param_col1:
        st.write("**权重向量 w:**")
        weights_df = pd.DataFrame(
            {
                "索引": [f"w[{i}]" for i in range(input_size)],
                "数值": [f"{w:.6f}" for w in neuron.weights],
            }
        )
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
        key="dense_upstream_grad",
        help="假设这是从损失函数传回的梯度。在真实训练中，这来自损失函数对输出的导数。",
    )

    st.markdown("---")

    # 执行完整的前向-反向-更新流程
    if st.button(
        "🚀 执行完整计算流程（前向→反向→更新）", type="primary", key="dense_compute"
    ):
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

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info('💡 根据项目定位，我们不仅展示"算了什么"，更要检测"什么时候会出问题"')

        # 收集所有检测结果
        stability_issues = []

        # 1. 检查梯度
        grad_check = StabilityChecker.check_gradient(gradients["weights"], "权重梯度")
        stability_issues.append(grad_check)

        bias_grad_check = StabilityChecker.check_gradient(
            np.array([gradients["bias"]]), "偏置梯度"
        )
        stability_issues.append(bias_grad_check)

        # 2. 检查激活值
        act_check = StabilityChecker.check_activation(
            np.array([neuron.forward_history["output"]]), "输出激活值"
        )
        stability_issues.append(act_check)

        # 3. 验证梯度正确性（数值梯度 vs 解析梯度）
        with st.spinner("正在计算数值梯度进行验证..."):
            try:
                numerical_grad = compute_numerical_gradient(
                    neuron, input_data, upstream_grad, epsilon=1e-5
                )
                grad_verify = StabilityChecker.verify_gradient(
                    numerical_grad, gradients["weights"], "权重梯度"
                )
                stability_issues.append(grad_verify)
            except Exception as e:
                st.warning(f"⚠️ 数值梯度计算失败: {str(e)}")

        # 4. 检查学习率
        param_norm = np.linalg.norm(neuron.weights)
        grad_norm = np.linalg.norm(gradients["weights"])
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, grad_norm, param_norm
        )
        stability_issues.append(lr_check)

        # 显示诊断结果
        StabilityChecker.display_issues(stability_issues)

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
            update_data.append(
                {
                    "参数": f"$w_{{{i}}}$",
                    "更新前": f"{old_weights[i]:.6f}",
                    "梯度": f'{gradients["weights"][i]:.6f}',
                    "更新量": f'{-learning_rate * gradients["weights"][i]:.6f}',
                    "更新后": f"{neuron.weights[i]:.6f}",
                    "变化": f"{delta:.6f}",
                }
            )

        delta_bias = neuron.bias - old_bias
        update_data.append(
            {
                "参数": "$b$",
                "更新前": f"{old_bias:.6f}",
                "梯度": f'{gradients["bias"]:.6f}',
                "更新量": f'{-learning_rate * gradients["bias"]:.6f}',
                "更新后": f"{neuron.bias:.6f}",
                "变化": f"{delta_bias:.6f}",
            }
        )

        update_df = pd.DataFrame(update_data)
        st.markdown(update_df.to_markdown(index=False))

        st.info(
            f"💡 **更新规则**: $\\theta_{{\text{{new}}}} = \\theta_{{\text{{old}}}} - \\alpha \\cdot \\frac{{\\partial L}}{{\\partial \\theta}}$，其中学习率 $\\alpha = {learning_rate}$"
        )

        # ==================== 总结 ====================
        st.markdown("---")
        st.subheader("📈 完整流程总结")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("前向传播输出", f"{output:.6f}", help="神经元的最终输出")

        with col2:
            avg_grad = np.mean(np.abs(gradients["weights"]))
            st.metric("平均权重梯度", f"{avg_grad:.6f}", help="权重梯度的平均绝对值")

        with col3:
            total_change = np.linalg.norm(neuron.weights - old_weights)
            st.metric("参数变化量", f"{total_change:.6f}", help="所有权重变化的L2范数")

        st.success(
            """
        ✅ **完整训练步骤已完成！**
        
        这就是神经网络训练的一个完整迭代：
        1. **前向传播**：输入 → 加权和 → 激活 → 输出
        2. **反向传播**：计算每个参数的梯度（链式法则）
        3. **参数更新**：沿着梯度的反方向更新参数
        
        在实际训练中，这个过程会重复成千上万次，每次使用不同的训练样本。
        """
        )


def render_conv_neuron():
    """渲染卷积神经元演示"""
    st.subheader("🖼️ 卷积神经元（Convolutional Neuron）")

    st.markdown(
        """
    **工作原理**：
    1. 使用小的卷积核扫描输入图像
    2. 局部连接：只关注输入的一小块区域
    3. 权值共享：同一个卷积核在整个输入上复用
    4. 提取局部特征（边缘、纹理等）
    
    **数学表达**：
    $$
    y = \\text{activation}\\left(\\sum_{i,j} K[i,j] \\cdot X[i,j] + b\\right)
    $$
    其中 $K$ 是卷积核，$X$ 是输入图像块
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置卷积神经元")

    col1, col2 = st.columns(2)

    with col1:
        kernel_size = st.slider(
            "卷积核大小", 2, 5, 3, help="卷积核的尺寸 (kernel_size × kernel_size)"
        )
        activation = st.selectbox(
            "激活函数",
            ["relu", "sigmoid", "tanh"],
            help="选择激活函数类型",
            key="conv_activation",
        )

    with col2:
        seed = st.number_input(
            "随机种子", 0, 100, 42, help="用于初始化卷积核", key="conv_seed"
        )
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长", key="conv_lr"
        )

    # 创建卷积神经元
    neuron = ConvNeuron(kernel_size=kernel_size, activation=activation, seed=seed)

    st.markdown("---")

    # 输入数据（图像块）
    st.subheader("📥 输入图像块")
    st.write(f"设置输入图像块（{kernel_size}×{kernel_size}）：")

    # 创建输入矩阵
    input_data = np.random.randn(kernel_size, kernel_size) * 0.5

    # 使用列显示输入
    for i in range(kernel_size):
        cols = st.columns(kernel_size)
        for j in range(kernel_size):
            with cols[j]:
                input_data[i, j] = st.number_input(
                    f"$X[{i},{j}]$",
                    value=float(input_data[i, j]),
                    format="%.4f",
                    key=f"conv_input_{i}_{j}",
                )

    # 显示卷积核
    st.subheader("🎯 卷积核（权重）")
    st.write("**卷积核 K:**")

    kernel_display = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel_display.append(
                {"位置": f"K[{i},{j}]", "数值": f"{neuron.kernel[i, j]:.6f}"}
            )

    kernel_df = pd.DataFrame(kernel_display)
    st.markdown(kernel_df.to_markdown(index=False))

    st.write(f"**偏置 b:** {neuron.bias:.6f}")

    st.markdown("---")

    # 上游梯度
    upstream_grad = st.number_input(
        "上游梯度 (∂L/∂y)",
        value=1.0,
        format="%.6f",
        key="conv_upstream_grad",
        help="来自损失函数的梯度",
    )

    st.markdown("---")

    # 执行计算
    if st.button("🚀 执行卷积计算", type="primary", key="conv_compute"):
        # 前向传播
        st.subheader("➡️ 1. 前向传播")
        output = neuron.forward(input_data)

        st.success(f"✅ 卷积输出: **{output:.6f}**")

        with st.expander("📋 详细计算步骤", expanded=True):
            # 显示逐元素乘积
            st.write("**逐元素相乘：**")
            element_wise = neuron.kernel * input_data

            steps = []
            for i in range(kernel_size):
                for j in range(kernel_size):
                    steps.append(
                        {
                            "位置": f"[{i},{j}]",
                            "卷积核": f"{neuron.kernel[i, j]:.4f}",
                            "输入": f"{input_data[i, j]:.4f}",
                            "乘积": f"{element_wise[i, j]:.4f}",
                        }
                    )

            steps_df = pd.DataFrame(steps)
            st.markdown(steps_df.to_markdown(index=False))

            st.write(f"**求和:** {np.sum(element_wise):.6f}")
            st.write(
                f"**加偏置:** {np.sum(element_wise):.6f} + {neuron.bias:.6f} = {neuron.forward_history['weighted_sum']:.6f}"
            )
            st.write(f"**激活函数 ({activation}):** {output:.6f}")

        # 可视化
        with st.expander("📊 卷积可视化", expanded=True):
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("输入图像块", "卷积核", "逐元素乘积"),
                specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
            )

            fig.add_trace(
                go.Heatmap(z=input_data, colorscale="Viridis", name="输入"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(z=neuron.kernel, colorscale="RdBu", name="卷积核"),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Heatmap(
                    z=neuron.kernel * input_data, colorscale="Greens", name="乘积"
                ),
                row=1,
                col=3,
            )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # 反向传播
        st.markdown("---")
        st.subheader("⬅️ 2. 反向传播")

        old_kernel = neuron.kernel.copy()
        old_bias = neuron.bias

        gradients = neuron.backward(upstream_grad)

        st.success("✅ 梯度计算完成")

        with st.expander("📋 梯度信息", expanded=True):
            grad_steps = []
            for i in range(kernel_size):
                for j in range(kernel_size):
                    grad_steps.append(
                        {
                            "位置": f"∂L/∂K[{i},{j}]",
                            "梯度": f'{gradients["kernel"][i, j]:.6f}',
                            "计算": f'{upstream_grad:.6f} × {neuron.forward_history["activation_derivative"]:.6f} × {input_data[i, j]:.6f}',
                        }
                    )

            grad_steps.append(
                {
                    "位置": "∂L/∂b",
                    "梯度": f'{gradients["bias"]:.6f}',
                    "计算": f'{upstream_grad:.6f} × {neuron.forward_history["activation_derivative"]:.6f}',
                }
            )

            grad_df = pd.DataFrame(grad_steps)
            st.markdown(grad_df.to_markdown(index=False))

        # 参数更新
        st.markdown("---")
        st.subheader("📊 3. 参数更新")

        neuron.update_parameters(learning_rate)

        st.success("✅ 参数已更新")

        st.write("**卷积核变化：**")
        kernel_change = neuron.kernel - old_kernel
        st.write(f"最大变化: {np.max(np.abs(kernel_change)):.6f}")
        st.write(f"平均变化: {np.mean(np.abs(kernel_change)):.6f}")

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info("💡 卷积特有问题：卷积核梯度、特征图爆炸、感受野过小")

        stability_issues = []

        # 1. 检查卷积核梯度
        grad_kernel_check = StabilityChecker.check_gradient(
            gradients["kernel"].flatten(), "卷积核梯度"
        )
        stability_issues.append(grad_kernel_check)

        grad_bias_check = StabilityChecker.check_gradient(
            np.array([gradients["bias"]]), "偏置梯度"
        )
        stability_issues.append(grad_bias_check)

        # 2. 检查输入和输出（特征图）
        input_check = StabilityChecker.check_activation(x_conv.flatten(), "输入图像块")
        stability_issues.append(input_check)

        output_check = StabilityChecker.check_activation(np.array([output]), "卷积输出")
        stability_issues.append(output_check)

        # 3. 检查卷积核的范数
        kernel_norm = np.linalg.norm(neuron.kernel)
        if kernel_norm > 10:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "卷积核范数过大",
                    "value": f"{kernel_norm:.4f}",
                    "threshold": "> 10",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "卷积核范数": f"{kernel_norm:.4f}",
                        "卷积核形状": f"{neuron.kernel.shape}",
                        "最大权重": f"{np.max(np.abs(neuron.kernel)):.4f}",
                    },
                    "solution": [
                        "使用Xavier/He初始化",
                        "添加权重衰减（L2正则化）",
                        "使用BatchNorm/LayerNorm",
                        "降低学习率",
                    ],
                    "explanation": "卷积核权重过大会导致输出爆炸，尤其在深层网络中累积",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "卷积核范数",
                    "value": f"{kernel_norm:.4f}",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "卷积核范数": f"{kernel_norm:.4f}",
                        "卷积核形状": f"{neuron.kernel.shape}",
                    },
                }
            )

        # 4. 检查卷积操作的数值范围
        element_wise_product = neuron.kernel * x_conv
        product_max = np.max(np.abs(element_wise_product))

        if product_max > 50:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "逐元素乘积过大",
                    "value": f"{product_max:.2f}",
                    "threshold": "> 50",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大乘积": f"{product_max:.2f}",
                        "求和结果": f"{np.sum(element_wise_product):.2f}",
                        "加偏置后": f"{np.sum(element_wise_product) + neuron.bias:.2f}",
                    },
                    "solution": [
                        "归一化输入（如除以255）",
                        "使用BatchNorm",
                        "减小卷积核权重（Xavier初始化）",
                    ],
                    "explanation": "输入与卷积核的乘积过大，可能导致后续激活函数饱和",
                }
            )

        # 5. 检查学习率
        param_norm = np.sqrt(np.linalg.norm(neuron.kernel) ** 2 + neuron.bias**2)
        grad_norm = np.sqrt(
            np.linalg.norm(gradients["kernel"]) ** 2 + gradients["bias"] ** 2
        )
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, grad_norm, param_norm
        )
        stability_issues.append(lr_check)

        # 6. 感受野分析
        receptive_field_size = kernel_size * kernel_size
        if kernel_size < 3:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "感受野过小",
                    "value": f"{kernel_size}×{kernel_size}",
                    "threshold": "< 3×3",
                    "icon": "🟡",
                    "severity": "low",
                    "details": {
                        "卷积核大小": f"{kernel_size}×{kernel_size}",
                        "感受野": f"{receptive_field_size}个像素",
                        "参数量": f"{kernel_size*kernel_size + 1}",
                    },
                    "solution": [
                        "使用3×3或更大的卷积核",
                        "堆叠多个小卷积核",
                        "使用空洞卷积增大感受野",
                    ],
                    "explanation": "1×1卷积无空间信息，2×2卷积感受野较小，3×3是常用选择",
                }
            )

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 卷积神经元稳定性诊断"
        )

        st.markdown("---")
        st.info(
            f"""
        💡 **卷积神经元特性与稳定性**：
        
        **局部连接**: 只处理 {kernel_size}×{kernel_size} 的输入区域
        - 感受野: {kernel_size*kernel_size} 个像素
        
        **权值共享**: 同一个卷积核可以扫描整个图像
        - 卷积核范数: {kernel_norm:.4f}
        - 参数效率: 相同参数处理整个图像
        
        **平移不变性**: 无论特征在哪里，都能被检测到
        - 输出范围: [{np.min(element_wise_product):.2f}, {np.max(element_wise_product):.2f}]
        
        **参数量**: {kernel_size}×{kernel_size} + 1 = {kernel_size*kernel_size + 1} 个参数
        - vs 全连接: 如果输入是28×28，全连接需要784个参数
        - 节省: {(1 - (kernel_size*kernel_size + 1) / 785) * 100:.1f}%
        
        **典型应用**:
        - 3×3: ResNet、VGG的标准选择
        - 5×5: AlexNet的第一层
        - 7×7: ResNet的输入层
        - 1×1: 通道数调整、降维
        """
        )


def render_rnn_neuron():
    """渲染循环神经元演示"""
    st.subheader("🔁 循环神经元（Recurrent Neural Network）")

    st.markdown(
        """
    **工作原理**：
    1. 维护一个隐藏状态 $h_t$，存储历史信息
    2. 当前状态由前一状态和当前输入共同决定
    3. 可以处理变长序列数据
    4. 具有"记忆"能力
    
    **数学表达**：
    $$
    h_t = \\text{activation}(W_{hh} h_{t-1} + W_{xh} x_t + b)
    $$
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置RNN神经元")

    col1, col2 = st.columns(2)

    with col1:
        input_size = st.slider(
            "输入维度", 1, 5, 3, help="每个时间步的输入大小", key="rnn_input_size"
        )
        hidden_size = st.slider(
            "隐藏层维度", 1, 5, 3, help="隐藏状态的大小", key="rnn_hidden_size"
        )
        activation = st.selectbox(
            "激活函数",
            ["tanh", "relu", "sigmoid"],
            help="RNN通常使用tanh",
            key="rnn_activation",
        )

    with col2:
        sequence_length = st.slider(
            "序列长度", 1, 5, 3, help="要处理的时间步数", key="rnn_seq_len"
        )
        seed = st.number_input(
            "随机种子", 0, 100, 42, help="用于初始化权重", key="rnn_seed"
        )
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长", key="rnn_lr"
        )

    # 创建RNN神经元
    neuron = RNNNeuron(
        input_size=input_size, hidden_size=hidden_size, activation=activation, seed=seed
    )

    st.markdown("---")

    # 输入序列
    st.subheader("📥 输入序列")
    st.write(f"设置 {sequence_length} 个时间步的输入：")

    sequence_data = []
    for t in range(sequence_length):
        st.write(f"**时间步 t={t}:**")
        time_step_input = []
        cols = st.columns(input_size)
        for i in range(input_size):
            with cols[i]:
                val = st.number_input(
                    f"$x_{t}[{i}]$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"rnn_input_t{t}_i{i}",
                )
                time_step_input.append(val)
        sequence_data.append(np.array(time_step_input))

    # 显示权重矩阵
    st.subheader("🎯 权重矩阵")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**W_xh (输入→隐藏):**")
        st.write(f"形状: ({hidden_size}, {input_size})")
        st.write(f"参数量: {hidden_size * input_size}")

    with col2:
        st.write("**W_hh (隐藏→隐藏):**")
        st.write(f"形状: ({hidden_size}, {hidden_size})")
        st.write(f"参数量: {hidden_size * hidden_size}")

    st.write(
        f"**总参数量**: {hidden_size * input_size + hidden_size * hidden_size + hidden_size}"
    )

    st.markdown("---")

    # 执行序列处理
    if st.button("🚀 处理序列", type="primary", key="rnn_compute"):
        st.subheader("➡️ 序列处理过程")

        # 重置隐藏状态
        neuron.reset_hidden()

        outputs = []
        hidden_states = [neuron.h.copy()]

        for t, x_t in enumerate(sequence_data):
            st.write(f"### 时间步 t={t}")

            h_t = neuron.forward(x_t)
            outputs.append(h_t.copy())
            hidden_states.append(h_t.copy())

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**输入:**")
                st.write(f"{x_t}")

            with col2:
                st.write("**前一状态:**")
                st.write(f"{hidden_states[t]}")

            with col3:
                st.write("**当前状态:**")
                st.write(f"{h_t}")

            st.write(f"**输出范数:** {np.linalg.norm(h_t):.6f}")
            st.markdown("---")

        # 可视化隐藏状态演化
        with st.expander("📊 隐藏状态演化", expanded=True):
            fig = go.Figure()

            for i in range(hidden_size):
                values = [h[i] for h in hidden_states]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(hidden_states))),
                        y=values,
                        mode="lines+markers",
                        name=f"h[{i}]",
                    )
                )

            fig.update_layout(
                title="隐藏状态随时间的变化",
                xaxis_title="时间步",
                yaxis_title="隐藏状态值",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # 反向传播（简化版，只展示最后一个时间步）
        st.markdown("---")
        st.subheader("⬅️ 反向传播（最后时间步）")

        upstream_grad = np.ones(hidden_size)
        st.write(f"上游梯度: {upstream_grad}")

        gradients = neuron.backward(upstream_grad)

        st.success("✅ 梯度计算完成")

        with st.expander("📋 梯度信息", expanded=True):
            st.write("**W_xh 梯度形状:**", gradients["W_xh"].shape)
            st.write("**W_xh 梯度范数:**", f"{np.linalg.norm(gradients['W_xh']):.6f}")
            st.write("**W_hh 梯度形状:**", gradients["W_hh"].shape)
            st.write("**W_hh 梯度范数:**", f"{np.linalg.norm(gradients['W_hh']):.6f}")
            st.write("**b_h 梯度:**", gradients["b_h"])

        # 参数更新
        st.markdown("---")
        st.subheader("📊 参数更新")

        old_W_xh = neuron.W_xh.copy()
        old_W_hh = neuron.W_hh.copy()

        neuron.update_parameters(learning_rate)

        st.success("✅ 参数已更新")

        change_xh = np.linalg.norm(neuron.W_xh - old_W_xh)
        change_hh = np.linalg.norm(neuron.W_hh - old_W_hh)

        st.write(f"**W_xh 变化量:** {change_xh:.6f}")
        st.write(f"**W_hh 变化量:** {change_hh:.6f}")

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info("💡 RNN特有问题：梯度消失（序列展开后）、隐藏状态爆炸")

        stability_issues = []

        # 1. 检查梯度
        grad_xh_check = StabilityChecker.check_gradient(
            gradients["W_xh"].flatten(), "W_xh权重梯度"
        )
        stability_issues.append(grad_xh_check)

        grad_hh_check = StabilityChecker.check_gradient(
            gradients["W_hh"].flatten(), "W_hh权重梯度"
        )
        stability_issues.append(grad_hh_check)

        # 2. 检查隐藏状态（关键！容易爆炸）
        for t, h in enumerate(hidden_states[1:]):  # 跳过初始状态
            h_check = StabilityChecker.check_activation(h, f"t={t}隐藏状态")
            if h_check["status"] != "success":
                stability_issues.append(h_check)

        # 如果所有隐藏状态都正常，添加一个总结
        if all(
            StabilityChecker.check_activation(h, "")["status"] == "success"
            for h in hidden_states[1:]
        ):
            stability_issues.append(
                {
                    "status": "success",
                    "type": "所有时间步隐藏状态",
                    "value": f"{len(hidden_states)-1}个状态",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "时间步数": len(hidden_states) - 1,
                        "最大范数": f"{max(np.linalg.norm(h) for h in hidden_states[1:]):.4f}",
                        "最小范数": f"{min(np.linalg.norm(h) for h in hidden_states[1:]):.4f}",
                    },
                }
            )

        # 3. 检查梯度通过时间的衰减（BPTT问题）
        if len(hidden_states) > 1:
            # 估计梯度衰减率
            first_h_norm = np.linalg.norm(hidden_states[1])
            last_h_norm = np.linalg.norm(hidden_states[-1])

            if last_h_norm > 0:
                decay_factor = first_h_norm / last_h_norm

                if decay_factor > 10:
                    stability_issues.append(
                        {
                            "status": "warning",
                            "type": "RNN梯度衰减",
                            "value": f"{decay_factor:.2f}倍",
                            "threshold": "> 10",
                            "icon": "🟡",
                            "severity": "medium",
                            "details": {
                                "首个状态范数": f"{first_h_norm:.4f}",
                                "最后状态范数": f"{last_h_norm:.4f}",
                                "衰减倍数": f"{decay_factor:.2f}",
                            },
                            "solution": [
                                "使用LSTM或GRU替代普通RNN",
                                "减少序列长度",
                                "使用梯度裁剪",
                                "使用更好的初始化（Orthogonal）",
                                "添加跳跃连接",
                            ],
                            "explanation": "RNN在长序列上容易出现梯度消失，LSTM/GRU通过门控机制解决这个问题",
                        }
                    )

        # 4. 检查W_hh的特征值（理论上应该接近1）
        eigenvalues = np.linalg.eigvals(neuron.W_hh)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        if max_eigenvalue > 1.1:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "W_hh特征值过大",
                    "value": f"{max_eigenvalue:.4f}",
                    "threshold": "> 1.1",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大特征值": f"{max_eigenvalue:.4f}",
                        "特征值范围": f"[{np.min(np.abs(eigenvalues)):.4f}, {max_eigenvalue:.4f}]",
                    },
                    "solution": [
                        "使用Orthogonal初始化",
                        "添加权重衰减",
                        "使用梯度裁剪",
                        "考虑使用LSTM/GRU",
                    ],
                    "explanation": "W_hh的最大特征值>1会导致梯度爆炸，<1会导致梯度消失",
                }
            )
        elif max_eigenvalue < 0.9:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "W_hh特征值过小",
                    "value": f"{max_eigenvalue:.4f}",
                    "threshold": "< 0.9",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大特征值": f"{max_eigenvalue:.4f}",
                        "特征值范围": f"[{np.min(np.abs(eigenvalues)):.4f}, {max_eigenvalue:.4f}]",
                    },
                    "solution": [
                        "使用Orthogonal初始化（特征值接近1）",
                        "增加学习率",
                        "考虑使用LSTM/GRU",
                    ],
                    "explanation": "W_hh的最大特征值<1会导致梯度消失，信号随时间衰减",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "W_hh特征值",
                    "value": f"{max_eigenvalue:.4f}",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "最大特征值": f"{max_eigenvalue:.4f}",
                        "理想范围": "[0.9, 1.1]",
                    },
                }
            )

        # 5. 检查学习率
        combined_grad_norm = np.sqrt(
            np.linalg.norm(gradients["W_xh"]) ** 2
            + np.linalg.norm(gradients["W_hh"]) ** 2
        )
        combined_param_norm = np.sqrt(
            np.linalg.norm(neuron.W_xh) ** 2 + np.linalg.norm(neuron.W_hh) ** 2
        )
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, combined_grad_norm, combined_param_norm
        )
        stability_issues.append(lr_check)

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 RNN神经元稳定性诊断"
        )

        # RNN特有的额外说明
        st.markdown("---")
        st.info(
            f"""
        💡 **RNN 特性与稳定性**：
        
        **记忆能力**: 隐藏状态 $h_t$ 存储历史信息
        - 当前最大范数: {max(np.linalg.norm(h) for h in hidden_states[1:]):.4f}
        
        **序列处理**: 可以处理任意长度的序列
        - 当前序列长度: {len(hidden_states)-1} 个时间步
        
        **参数共享**: 所有时间步使用相同的权重
        - W_hh特征值: {max_eigenvalue:.4f} (理想≈1.0)
        
        **时间展开**: 反向传播需要沿时间展开（BPTT）
        - 梯度需要回传 {len(hidden_states)-1} 个时间步
        
        **梯度问题**: 长序列可能出现梯度消失或爆炸
        - 建议: 序列长度<50，或使用LSTM/GRU
        
        **总参数量**: {hidden_size * input_size + hidden_size * hidden_size + hidden_size} 个
        """
        )


def render_gru_neuron():
    """渲染GRU神经元演示"""
    st.subheader("🔄 GRU神经元（Gated Recurrent Unit）")

    st.markdown(
        """
    **工作原理**：
    1. 使用2个门控制信息流动（比LSTM少1个门）
    2. 重置门决定如何结合新输入和前一状态
    3. 更新门决定保留多少前一状态
    4. 没有单独的细胞状态，直接更新隐藏状态
    
    **数学表达**：
    $$
    \\begin{aligned}
    r_t &= \\sigma(W_r \\cdot [h_{t-1}, x_t] + b_r) && \\text{重置门} \\\\
    z_t &= \\sigma(W_z \\cdot [h_{t-1}, x_t] + b_z) && \\text{更新门} \\\\
    \\tilde{h}_t &= \\tanh(W_h \\cdot [r_t \\odot h_{t-1}, x_t] + b_h) && \\text{候选状态} \\\\
    h_t &= (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t && \\text{最终状态}
    \\end{aligned}
    $$
    
    **GRU vs LSTM vs RNN**：
    - **RNN**: 无门控，梯度消失严重
    - **LSTM**: 3个门 + 细胞状态，参数最多，能力最强
    - **GRU**: 2个门，参数适中，性能接近LSTM但更快
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置GRU神经元")

    col1, col2 = st.columns(2)

    with col1:
        input_size = st.slider(
            "输入维度", 1, 5, 3, help="每个时间步的输入大小", key="gru_input_size"
        )
        hidden_size = st.slider(
            "隐藏层维度", 1, 5, 3, help="隐藏状态的大小", key="gru_hidden_size"
        )

    with col2:
        sequence_length = st.slider(
            "序列长度", 1, 5, 3, help="要处理的时间步数", key="gru_seq_len"
        )
        seed = st.number_input(
            "随机种子", 0, 100, 42, help="用于初始化权重", key="gru_seed"
        )
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长", key="gru_lr"
        )

    # 创建GRU神经元
    neuron = GRUNeuron(input_size=input_size, hidden_size=hidden_size, seed=seed)

    st.markdown("---")

    # 显示参数量并与RNN、LSTM对比
    st.subheader("🎯 参数量分析与对比")
    combined_size = hidden_size + input_size
    params_per_gate = hidden_size * combined_size + hidden_size
    gru_total_params = 3 * params_per_gate

    # 计算对比数据
    rnn_params = hidden_size * input_size + hidden_size * hidden_size + hidden_size
    lstm_params = 4 * params_per_gate

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RNN参数量", f"{rnn_params}", delta=f"基准", delta_color="off")
    with col2:
        st.metric(
            "GRU参数量",
            f"{gru_total_params}",
            delta=f"+{gru_total_params - rnn_params} vs RNN",
        )
    with col3:
        st.metric(
            "LSTM参数量",
            f"{lstm_params}",
            delta=f"+{lstm_params - gru_total_params} vs GRU",
        )

    st.info(
        f"""
    💡 **参数量对比**（input={input_size}, hidden={hidden_size}）：
    - **RNN**: {rnn_params}个参数 - 最少，但梯度消失严重
    - **GRU**: {gru_total_params}个参数 - 适中，性能好且快速
    - **LSTM**: {lstm_params}个参数 - 最多，能力最强
    
    **GRU的优势**：
    - ✅ 比LSTM少25%参数（3个门 vs 4个门）
    - ✅ 训练和推理速度更快
    - ✅ 在很多任务上性能与LSTM相当
    - ✅ 更简单，更容易理解和调试
    """
    )

    st.markdown("---")

    # 输入序列
    st.subheader("📥 输入序列")
    st.write(f"设置 {sequence_length} 个时间步的输入：")

    sequence_data = []
    for t in range(sequence_length):
        st.write(f"**时间步 t={t}:**")
        time_step_input = []
        cols = st.columns(input_size)
        for i in range(input_size):
            with cols[i]:
                val = st.number_input(
                    f"$x_{t}[{i}]$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"gru_input_t{t}_i{i}",
                )
                time_step_input.append(val)
        sequence_data.append(np.array(time_step_input))

    st.markdown("---")

    # 执行序列处理
    if st.button("🚀 处理GRU序列", type="primary", key="gru_compute"):
        st.subheader("➡️ 序列处理过程")

        # 重置隐藏状态
        neuron.reset_state()

        outputs = []
        hidden_states = [neuron.h.copy()]
        gate_history = []

        for t, x_t in enumerate(sequence_data):
            st.write(f"### 时间步 t={t}")

            h_t = neuron.forward(x_t)
            outputs.append(h_t.copy())
            hidden_states.append(h_t.copy())

            # 保存门的值
            gate_history.append(
                {
                    "r_t": neuron.forward_history["r_t"].copy(),
                    "z_t": neuron.forward_history["z_t"].copy(),
                    "h_tilde": neuron.forward_history["h_tilde"].copy(),
                }
            )

            # 显示详细计算过程（体现项目定位）
            with st.expander(f"🔬 t={t} 的详细计算步骤", expanded=True):
                st.write("**步骤1: 拼接输入**")
                st.write(f"$[h_{{t-1}}, x_t] = {neuron.forward_history['combined']}$")

                st.write("**步骤2: 计算重置门（决定如何使用历史信息）**")
                st.write(f"$r_t = \\sigma(W_r \\cdot [h_{{t-1}}, x_t] + b_r)$")
                gate_data = []
                for i in range(hidden_size):
                    gate_data.append(
                        {
                            "维度": f"[{i}]",
                            "重置门值": f'{neuron.forward_history["r_t"][i]:.4f}',
                            "含义": (
                                "接近0→忽略历史"
                                if neuron.forward_history["r_t"][i] < 0.5
                                else "接近1→保留历史"
                            ),
                        }
                    )
                gate_df = pd.DataFrame(gate_data)
                st.markdown(gate_df.to_markdown(index=False))

                st.write("**步骤3: 应用重置门**")
                st.write(
                    f"重置后的历史: $r_t \\odot h_{{t-1}} = {neuron.forward_history['r_t'] * neuron.forward_history['h_prev']}$"
                )

                st.write("**步骤4: 计算候选隐藏状态**")
                st.write(
                    f"$\\tilde{{h}}_t = \\tanh(W_h \\cdot [r_t \\odot h_{{t-1}}, x_t] + b_h)$"
                )
                st.write(f"$\\tilde{{h}}_t = {neuron.forward_history['h_tilde']}$")

                st.write("**步骤5: 计算更新门（决定新旧信息的比例）**")
                st.write(f"$z_t = \\sigma(W_z \\cdot [h_{{t-1}}, x_t] + b_z)$")
                update_data = []
                for i in range(hidden_size):
                    update_data.append(
                        {
                            "维度": f"[{i}]",
                            "更新门值": f'{neuron.forward_history["z_t"][i]:.4f}',
                            "旧信息权重": f'{1 - neuron.forward_history["z_t"][i]:.4f}',
                            "新信息权重": f'{neuron.forward_history["z_t"][i]:.4f}',
                        }
                    )
                update_df = pd.DataFrame(update_data)
                st.markdown(update_df.to_markdown(index=False))

                st.write("**步骤6: 计算最终隐藏状态（加权组合）**")
                st.write(
                    f"$h_t = (1 - z_t) \\odot h_{{t-1}} + z_t \\odot \\tilde{{h}}_t$"
                )

                final_data = []
                for i in range(hidden_size):
                    old_contrib = (
                        1 - neuron.forward_history["z_t"][i]
                    ) * neuron.forward_history["h_prev"][i]
                    new_contrib = (
                        neuron.forward_history["z_t"][i]
                        * neuron.forward_history["h_tilde"][i]
                    )
                    final_data.append(
                        {
                            "维度": f"[{i}]",
                            "旧状态贡献": f"{old_contrib:.4f}",
                            "新状态贡献": f"{new_contrib:.4f}",
                            "最终值": f"{h_t[i]:.4f}",
                        }
                    )
                final_df = pd.DataFrame(final_data)
                st.markdown(final_df.to_markdown(index=False))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("隐藏状态范数", f"{np.linalg.norm(h_t):.4f}")
            with col2:
                avg_reset = np.mean(neuron.forward_history["r_t"])
                st.metric("平均重置率", f"{avg_reset:.4f}")
            with col3:
                avg_update = np.mean(neuron.forward_history["z_t"])
                st.metric("平均更新率", f"{avg_update:.4f}")
            with col4:
                # 计算信息保留率
                retention = 1 - avg_update
                st.metric("信息保留率", f"{retention:.4f}")

            st.markdown("---")

        # 可视化门的演化
        with st.expander("📊 门控值演化", expanded=True):
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "重置门 r_t",
                    "更新门 z_t",
                    "候选状态 h̃_t",
                    "隐藏状态范数",
                ),
            )

            for i in range(hidden_size):
                # 重置门
                r_values = [g["r_t"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(r_values))),
                        y=r_values,
                        mode="lines+markers",
                        name=f"r[{i}]",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

                # 更新门
                z_values = [g["z_t"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(z_values))),
                        y=z_values,
                        mode="lines+markers",
                        name=f"z[{i}]",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

                # 候选状态
                h_tilde_values = [g["h_tilde"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(h_tilde_values))),
                        y=h_tilde_values,
                        mode="lines+markers",
                        name=f"h̃[{i}]",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            # 隐藏状态范数
            h_norms = [np.linalg.norm(h) for h in hidden_states]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(h_norms))),
                    y=h_norms,
                    mode="lines+markers",
                    name="||h||",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

            fig.update_xaxes(title_text="时间步", row=2, col=1)
            fig.update_xaxes(title_text="时间步", row=2, col=2)
            fig.update_yaxes(title_text="门值", row=1, col=1)
            fig.update_yaxes(title_text="门值", row=1, col=2)
            fig.update_yaxes(title_text="候选值", row=2, col=1)
            fig.update_yaxes(title_text="范数", row=2, col=2)

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # 反向传播
        st.markdown("---")
        st.subheader("⬅️ 反向传播（最后时间步）")

        upstream_grad = np.ones(hidden_size)
        st.write(f"上游梯度: {upstream_grad}")

        gradients = neuron.backward(upstream_grad)

        st.success("✅ 梯度计算完成")

        with st.expander("📋 梯度信息", expanded=True):
            grad_info = []
            for gate_name in ["r", "z", "h"]:
                W_key = f"grad_W_{gate_name}"
                b_key = f"grad_b_{gate_name}"
                grad_info.append(
                    {
                        "门/状态": (
                            f"{gate_name}门" if gate_name in ["r", "z"] else "候选状态"
                        ),
                        "W梯度范数": f"{np.linalg.norm(gradients[W_key]):.6f}",
                        "b梯度范数": f"{np.linalg.norm(gradients[b_key]):.6f}",
                    }
                )

            grad_df = pd.DataFrame(grad_info)
            st.markdown(grad_df.to_markdown(index=False))

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info("💡 GRU特有检测：门控饱和、梯度消失、序列长度影响")

        stability_issues = []

        # 1. 检查梯度
        for gate_name in ["r", "z", "h"]:
            W_key = f"grad_W_{gate_name}"
            gate_label = {"r": "重置门", "z": "更新门", "h": "候选状态"}[gate_name]
            grad_check = StabilityChecker.check_gradient(
                gradients[W_key].flatten(), f"{gate_label}权重梯度"
            )
            stability_issues.append(grad_check)

        # 2. 检查门控饱和（GRU特有）
        if gate_history:
            last_gates = gate_history[-1]

            # 检查重置门
            r_check = StabilityChecker.check_gate_saturation(
                last_gates["r_t"], "重置门 r_t"
            )
            stability_issues.append(r_check)

            # 检查更新门
            z_check = StabilityChecker.check_gate_saturation(
                last_gates["z_t"], "更新门 z_t"
            )
            stability_issues.append(z_check)

        # 3. 检查隐藏状态
        h_check = StabilityChecker.check_activation(neuron.h, "隐藏状态")
        stability_issues.append(h_check)

        # 4. 检查学习率
        combined_grad_norm = np.sqrt(
            np.linalg.norm(gradients["grad_W_r"]) ** 2
            + np.linalg.norm(gradients["grad_W_z"]) ** 2
            + np.linalg.norm(gradients["grad_W_h"]) ** 2
        )
        combined_param_norm = np.sqrt(
            np.linalg.norm(neuron.W_r) ** 2
            + np.linalg.norm(neuron.W_z) ** 2
            + np.linalg.norm(neuron.W_h) ** 2
        )
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, combined_grad_norm, combined_param_norm
        )
        stability_issues.append(lr_check)

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 GRU神经元稳定性诊断"
        )

        # 参数更新
        st.markdown("---")
        st.subheader("📊 参数更新")

        old_W_r = neuron.W_r.copy()
        old_W_z = neuron.W_z.copy()
        old_W_h = neuron.W_h.copy()

        neuron.update_parameters(learning_rate)

        st.success("✅ 参数已更新")

        change_r = np.linalg.norm(neuron.W_r - old_W_r)
        change_z = np.linalg.norm(neuron.W_z - old_W_z)
        change_h = np.linalg.norm(neuron.W_h - old_W_h)

        st.write("**各门权重变化量：**")
        changes_data = [
            {"门/状态": "重置门", "变化量": f"{change_r:.6f}"},
            {"门/状态": "更新门", "变化量": f"{change_z:.6f}"},
            {"门/状态": "候选状态", "变化量": f"{change_h:.6f}"},
        ]
        changes_df = pd.DataFrame(changes_data)
        st.markdown(changes_df.to_markdown(index=False))

        st.info(
            f"""
        💡 **GRU 特性总结**：
        
        **计算细节**（项目核心）：
        - 每个时间步进行6次矩阵运算
        - 2次sigmoid激活（门控）
        - 1次tanh激活（候选状态）
        - 3次逐元素乘法（门控应用）
        
        **门控机制**：
        - **重置门** $r_t$: 控制如何结合历史信息（接近0→忽略历史）
        - **更新门** $z_t$: 控制新旧信息比例（接近1→采用新信息）
        
        **优势对比**：
        - vs RNN: 解决梯度消失，能捕获长期依赖
        - vs LSTM: 参数少25%，速度快，性能相当
        
        **实际应用**：
        - 机器翻译（Google Translate用过GRU）
        - 语音识别
        - 文本生成
        - 时序预测
        
        **总参数量**: {gru_total_params} 个
        **浮点运算数**: 约 {3 * hidden_size * (hidden_size + input_size) * 2} FLOPs/时间步
        """
        )


def render_lstm_neuron():
    """渲染LSTM神经元演示"""
    st.subheader("🧠 LSTM神经元（Long Short-Term Memory）")

    st.markdown(
        """
    **工作原理**：
    1. 使用门控机制控制信息流动
    2. 维护细胞状态 $C_t$ 存储长期记忆
    3. 三个门：遗忘门、输入门、输出门
    4. 解决RNN的梯度消失问题
    
    **数学表达**（门控机制）：
    $$
    \\begin{aligned}
    f_t &= \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) && \\text{遗忘门} \\\\
    i_t &= \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) && \\text{输入门} \\\\
    \\tilde{C}_t &= \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C) && \\text{候选值} \\\\
    C_t &= f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t && \\text{更新细胞} \\\\
    o_t &= \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) && \\text{输出门} \\\\
    h_t &= o_t \\odot \\tanh(C_t) && \\text{输出}
    \\end{aligned}
    $$
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置LSTM神经元")

    col1, col2 = st.columns(2)

    with col1:
        input_size = st.slider(
            "输入维度", 1, 5, 3, help="每个时间步的输入大小", key="lstm_input_size"
        )
        hidden_size = st.slider(
            "隐藏层维度",
            1,
            5,
            3,
            help="隐藏状态和细胞状态的大小",
            key="lstm_hidden_size",
        )

    with col2:
        sequence_length = st.slider(
            "序列长度", 1, 5, 3, help="要处理的时间步数", key="lstm_seq_len"
        )
        seed = st.number_input(
            "随机种子", 0, 100, 42, help="用于初始化权重", key="lstm_seed"
        )
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长", key="lstm_lr"
        )

    # 创建LSTM神经元
    neuron = LSTMNeuron(input_size=input_size, hidden_size=hidden_size, seed=seed)

    st.markdown("---")

    # 显示参数量
    st.subheader("🎯 参数量分析")
    combined_size = hidden_size + input_size
    params_per_gate = hidden_size * combined_size + hidden_size
    total_params = 4 * params_per_gate

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("每个门的参数", f"{params_per_gate}")
    with col2:
        st.metric("门的数量", "4")
    with col3:
        st.metric("总参数量", f"{total_params}")

    st.info(
        """
    💡 **LSTM有4组权重**：
    - 遗忘门 $W_f, b_f$：决定遗忘多少历史信息
    - 输入门 $W_i, b_i$：决定接受多少新信息
    - 候选值 $W_C, b_C$：生成候选记忆
    - 输出门 $W_o, b_o$：决定输出多少信息
    """
    )

    st.markdown("---")

    # 输入序列
    st.subheader("📥 输入序列")
    st.write(f"设置 {sequence_length} 个时间步的输入：")

    sequence_data = []
    for t in range(sequence_length):
        st.write(f"**时间步 t={t}:**")
        time_step_input = []
        cols = st.columns(input_size)
        for i in range(input_size):
            with cols[i]:
                val = st.number_input(
                    f"$x_{t}[{i}]$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"lstm_input_t{t}_i{i}",
                )
                time_step_input.append(val)
        sequence_data.append(np.array(time_step_input))

    st.markdown("---")

    # 执行序列处理
    if st.button("🚀 处理LSTM序列", type="primary", key="lstm_compute"):
        st.subheader("➡️ 序列处理过程")

        # 重置状态
        neuron.reset_state()

        outputs = []
        hidden_states = [neuron.h.copy()]
        cell_states = [neuron.C.copy()]
        gate_history = []

        for t, x_t in enumerate(sequence_data):
            st.write(f"### 时间步 t={t}")

            h_t = neuron.forward(x_t)
            outputs.append(h_t.copy())
            hidden_states.append(h_t.copy())
            cell_states.append(neuron.C.copy())

            # 保存门的值
            gate_history.append(
                {
                    "f_t": neuron.forward_history["f_t"].copy(),
                    "i_t": neuron.forward_history["i_t"].copy(),
                    "o_t": neuron.forward_history["o_t"].copy(),
                    "C_tilde": neuron.forward_history["C_tilde"].copy(),
                }
            )

            # 显示门的激活值
            with st.expander(f"🚪 t={t} 的门控值", expanded=False):
                gate_data = []
                for i in range(hidden_size):
                    gate_data.append(
                        {
                            "维度": f"[{i}]",
                            "遗忘门": f'{neuron.forward_history["f_t"][i]:.4f}',
                            "输入门": f'{neuron.forward_history["i_t"][i]:.4f}',
                            "输出门": f'{neuron.forward_history["o_t"][i]:.4f}',
                            "候选值": f'{neuron.forward_history["C_tilde"][i]:.4f}',
                        }
                    )

                gate_df = pd.DataFrame(gate_data)
                st.markdown(gate_df.to_markdown(index=False))

                st.write(f"**细胞状态 $C_t$:** {neuron.C}")
                st.write(f"**隐藏状态 $h_t$:** {h_t}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("细胞状态范数", f"{np.linalg.norm(neuron.C):.4f}")
            with col2:
                st.metric("隐藏状态范数", f"{np.linalg.norm(h_t):.4f}")
            with col3:
                avg_forget = np.mean(neuron.forward_history["f_t"])
                st.metric("平均遗忘率", f"{avg_forget:.4f}")

            st.markdown("---")

        # 可视化门的演化
        with st.expander("📊 门控值演化", expanded=True):
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("遗忘门", "输入门", "输出门", "细胞状态范数"),
            )

            for i in range(hidden_size):
                # 遗忘门
                f_values = [g["f_t"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(f_values))),
                        y=f_values,
                        mode="lines+markers",
                        name=f"f[{i}]",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

                # 输入门
                i_values = [g["i_t"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(i_values))),
                        y=i_values,
                        mode="lines+markers",
                        name=f"i[{i}]",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

                # 输出门
                o_values = [g["o_t"][i] for g in gate_history]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(o_values))),
                        y=o_values,
                        mode="lines+markers",
                        name=f"o[{i}]",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            # 细胞状态范数
            c_norms = [np.linalg.norm(c) for c in cell_states]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(c_norms))),
                    y=c_norms,
                    mode="lines+markers",
                    name="||C||",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

            fig.update_xaxes(title_text="时间步", row=2, col=1)
            fig.update_xaxes(title_text="时间步", row=2, col=2)
            fig.update_yaxes(title_text="门值", row=1, col=1)
            fig.update_yaxes(title_text="门值", row=1, col=2)
            fig.update_yaxes(title_text="门值", row=2, col=1)
            fig.update_yaxes(title_text="范数", row=2, col=2)

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # 反向传播
        st.markdown("---")
        st.subheader("⬅️ 反向传播（最后时间步）")

        upstream_grad = np.ones(hidden_size)
        st.write(f"上游梯度: {upstream_grad}")

        gradients = neuron.backward(upstream_grad)

        st.success("✅ 梯度计算完成")

        with st.expander("📋 梯度信息", expanded=True):
            grad_info = []
            for gate_name in ["f", "i", "C", "o"]:
                W_key = f"grad_W_{gate_name}"
                b_key = f"grad_b_{gate_name}"
                grad_info.append(
                    {
                        "门": f"{gate_name}门",
                        "W梯度范数": f"{np.linalg.norm(gradients[W_key]):.6f}",
                        "b梯度范数": f"{np.linalg.norm(gradients[b_key]):.6f}",
                    }
                )

            grad_df = pd.DataFrame(grad_info)
            st.markdown(grad_df.to_markdown(index=False))

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info("💡 LSTM特有检测：4个门控饱和、细胞状态爆炸、长期依赖问题")

        stability_issues = []

        # 1. 检查梯度
        gate_labels = {"f": "遗忘门", "i": "输入门", "C": "候选值", "o": "输出门"}
        for gate_name in ["f", "i", "C", "o"]:
            W_key = f"grad_W_{gate_name}"
            grad_check = StabilityChecker.check_gradient(
                gradients[W_key].flatten(), f"{gate_labels[gate_name]}权重梯度"
            )
            stability_issues.append(grad_check)

        # 2. 检查门控饱和（LSTM特有 - 4个门）
        if gate_history:
            last_gates = gate_history[-1]

            # 检查遗忘门
            f_check = StabilityChecker.check_gate_saturation(
                last_gates["f_t"], "遗忘门 f_t"
            )
            stability_issues.append(f_check)

            # 检查输入门
            i_check = StabilityChecker.check_gate_saturation(
                last_gates["i_t"], "输入门 i_t"
            )
            stability_issues.append(i_check)

            # 检查输出门
            o_check = StabilityChecker.check_gate_saturation(
                last_gates["o_t"], "输出门 o_t"
            )
            stability_issues.append(o_check)

        # 3. 检查细胞状态和隐藏状态
        c_check = StabilityChecker.check_activation(neuron.C, "细胞状态 C_t")
        stability_issues.append(c_check)

        h_check = StabilityChecker.check_activation(neuron.h, "隐藏状态 h_t")
        stability_issues.append(h_check)

        # 4. 检查学习率
        combined_grad_norm = np.sqrt(
            sum(
                [
                    np.linalg.norm(gradients[f"grad_W_{g}"]) ** 2
                    for g in ["f", "i", "C", "o"]
                ]
            )
        )
        combined_param_norm = np.sqrt(
            sum(
                [
                    np.linalg.norm(getattr(neuron, f"W_{g}")) ** 2
                    for g in ["f", "i", "C", "o"]
                ]
            )
        )
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, combined_grad_norm, combined_param_norm
        )
        stability_issues.append(lr_check)

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 LSTM神经元稳定性诊断"
        )

        # 参数更新
        st.markdown("---")
        st.subheader("📊 参数更新")

        old_W_f = neuron.W_f.copy()
        old_W_i = neuron.W_i.copy()
        old_W_C = neuron.W_C.copy()
        old_W_o = neuron.W_o.copy()

        neuron.update_parameters(learning_rate)

        st.success("✅ 参数已更新")

        change_f = np.linalg.norm(neuron.W_f - old_W_f)
        change_i = np.linalg.norm(neuron.W_i - old_W_i)
        change_C = np.linalg.norm(neuron.W_C - old_W_C)
        change_o = np.linalg.norm(neuron.W_o - old_W_o)

        st.write("**各门权重变化量：**")
        changes_data = [
            {"门": "遗忘门", "变化量": f"{change_f:.6f}"},
            {"门": "输入门", "变化量": f"{change_i:.6f}"},
            {"门": "候选值", "变化量": f"{change_C:.6f}"},
            {"门": "输出门", "变化量": f"{change_o:.6f}"},
        ]
        changes_df = pd.DataFrame(changes_data)
        st.markdown(changes_df.to_markdown(index=False))

        st.info(
            f"""
        💡 **LSTM 特性**：
        - **门控机制**: 精确控制信息流动
        - **细胞状态**: 长期记忆通道，梯度可以直接流动
        - **解决梯度消失**: 通过加法操作而非乘法，梯度更容易回传
        - **参数量**: 是普通RNN的4倍，但效果显著提升
        - **应用**: 机器翻译、语音识别、文本生成
        
        **总参数量**: {total_params} 个
        """
        )


def render_attention_neuron():
    """渲染注意力机制神经元演示"""
    st.subheader("🎯 注意力机制（Attention Mechanism）")

    st.markdown(
        """
    **工作原理**：
    1. Query（查询）、Key（键）、Value（值）三个角色
    2. 计算Query与每个Key的相似度（注意力分数）
    3. 用Softmax归一化得到注意力权重
    4. 对Value进行加权求和
    
    **数学表达**：
    $$
    \\begin{aligned}
    Q &= x_q \\cdot W_Q \\\\
    K_i &= x_{k,i} \\cdot W_K \\\\
    V_i &= x_{v,i} \\cdot W_V \\\\
    \\text{score}_i &= \\frac{Q \\cdot K_i^T}{\\sqrt{d_k}} \\\\
    \\alpha_i &= \\frac{\\exp(\\text{score}_i)}{\\sum_j \\exp(\\text{score}_j)} \\\\
    \\text{output} &= \\sum_i \\alpha_i \\cdot V_i
    \\end{aligned}
    $$
    """
    )

    st.markdown("---")
    st.subheader("⚙️ 配置注意力机制")

    col1, col2 = st.columns(2)

    with col1:
        input_size = st.slider(
            "输入维度", 2, 5, 3, help="输入向量的维度", key="attn_input_size"
        )
        d_model = st.slider(
            "模型维度", 2, 5, 3, help="Q、K、V的维度", key="attn_d_model"
        )

    with col2:
        num_keys = st.slider(
            "Key/Value数量", 2, 5, 3, help="要注意的位置数量", key="attn_num_keys"
        )
        seed = st.number_input(
            "随机种子", 0, 100, 42, help="用于初始化权重", key="attn_seed"
        )
        learning_rate = st.slider(
            "学习率", 0.001, 0.5, 0.01, 0.001, help="梯度下降的步长", key="attn_lr"
        )

    # 创建注意力神经元
    neuron = AttentionNeuron(input_size=input_size, d_model=d_model, seed=seed)

    st.markdown("---")

    # 参数量分析
    st.subheader("🎯 参数量分析")
    total_params = 3 * d_model * input_size

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("W_Q 参数", f"{d_model * input_size}")
    with col2:
        st.metric("W_K 参数", f"{d_model * input_size}")
    with col3:
        st.metric("W_V 参数", f"{d_model * input_size}")

    st.info(
        f"""
    💡 **注意力机制的参数**：
    - $W_Q$: 将输入投影到Query空间
    - $W_K$: 将输入投影到Key空间
    - $W_V$: 将输入投影到Value空间
    - **总参数量**: {total_params}
    - **注意力权重**是动态计算的，不是固定参数！
    """
    )

    st.markdown("---")

    # 输入数据
    st.subheader("📥 输入数据")

    st.write("**Query（查询向量）：**")
    query_cols = st.columns(input_size)
    query_data = []
    for i in range(input_size):
        with query_cols[i]:
            val = st.number_input(
                f"$q[{i}]$",
                value=float(np.random.randn() * 0.5),
                format="%.4f",
                key=f"attn_query_{i}",
            )
            query_data.append(val)
    query_data = np.array(query_data)

    st.write(f"**Keys & Values（{num_keys}个键值对）：**")
    keys_data = []
    values_data = []

    for k in range(num_keys):
        st.write(f"*位置 {k}:*")
        key_cols = st.columns(input_size)
        key_row = []
        for i in range(input_size):
            with key_cols[i]:
                val = st.number_input(
                    f"$k_{k}[{i}]$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"attn_key_{k}_{i}",
                )
                key_row.append(val)
        keys_data.append(np.array(key_row))

        # Value使用与Key相同的数据（简化）
        value_cols = st.columns(input_size)
        value_row = []
        for i in range(input_size):
            with value_cols[i]:
                val = st.number_input(
                    f"$v_{k}[{i}]$",
                    value=float(np.random.randn() * 0.5),
                    format="%.4f",
                    key=f"attn_value_{k}_{i}",
                )
                value_row.append(val)
        values_data.append(np.array(value_row))

    keys_data = np.array(keys_data)
    values_data = np.array(values_data)

    st.markdown("---")

    # 执行注意力计算
    if st.button("🚀 计算注意力", type="primary", key="attn_compute"):
        st.subheader("➡️ 1. 前向传播")

        output, attention_weights = neuron.forward(query_data, keys_data, values_data)

        st.success(f"✅ 注意力输出: {output}")

        # 详细计算步骤
        with st.expander("📋 详细计算步骤", expanded=True):
            st.write("**步骤1: 投影到Q、K、V空间**")
            st.write(f"Query: $Q = W_Q \\cdot q$")
            st.write(f"$Q = {neuron.forward_history['Q']}$")
            st.write(f"Keys: $K_i = W_K \\cdot k_i$")
            for i, k in enumerate(neuron.forward_history["K"]):
                st.write(f"$K_{i} = {k}$")

            st.write("**步骤2: 计算注意力分数**")
            st.write(
                f"$\\text{{score}}_i = \\frac{{Q \\cdot K_i^T}}{{\\sqrt{{{d_model}}}}}$"
            )
            scores_data = []
            for i, score in enumerate(neuron.forward_history["scores"]):
                scores_data.append(
                    {
                        "位置": f"{i}",
                        "原始分数": f"{score * np.sqrt(d_model):.4f}",
                        "缩放后": f"{score:.4f}",
                    }
                )
            scores_df = pd.DataFrame(scores_data)
            st.markdown(scores_df.to_markdown(index=False))

            st.write("**步骤3: Softmax归一化**")
            attn_data = []
            for i, weight in enumerate(attention_weights):
                attn_data.append(
                    {
                        "位置": f"{i}",
                        "注意力权重": f"{weight:.4f}",
                        "百分比": f"{weight*100:.2f}%",
                    }
                )
            attn_df = pd.DataFrame(attn_data)
            st.markdown(attn_df.to_markdown(index=False))
            st.write(f"权重和: {np.sum(attention_weights):.6f} (应该=1.0)")

            st.write("**步骤4: 加权求和Value**")
            st.write(f"$\\text{{output}} = \\sum_i \\alpha_i \\cdot V_i$")
            for i, (w, v) in enumerate(
                zip(attention_weights, neuron.forward_history["V"])
            ):
                st.write(f"$\\alpha_{i} \\cdot V_{i} = {w:.4f} \\times {v} = {w * v}$")
            st.write(f"**最终输出**: {output}")

        # 可视化注意力权重
        with st.expander("📊 注意力权重可视化", expanded=True):
            fig = go.Figure()

            # 条形图
            fig.add_trace(
                go.Bar(
                    x=[f"位置{i}" for i in range(num_keys)],
                    y=attention_weights,
                    text=[f"{w:.3f}" for w in attention_weights],
                    textposition="auto",
                    marker=dict(
                        color=attention_weights,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="权重"),
                    ),
                )
            )

            fig.update_layout(
                title="注意力权重分布",
                xaxis_title="Key/Value位置",
                yaxis_title="注意力权重",
                height=400,
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 热力图：展示Query与每个Key的相似度
            fig2 = go.Figure(
                data=go.Heatmap(
                    z=[attention_weights],
                    x=[f"K{i}" for i in range(num_keys)],
                    y=["Query"],
                    colorscale="RdYlGn",
                    text=[[f"{w:.3f}" for w in attention_weights]],
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorbar=dict(title="注意力"),
                )
            )

            fig2.update_layout(title="Query对各Key的注意力", height=200)
            st.plotly_chart(fig2, use_container_width=True)

        # 反向传播
        st.markdown("---")
        st.subheader("⬅️ 2. 反向传播")

        upstream_grad = np.ones(d_model)
        st.write(f"上游梯度: {upstream_grad}")

        gradients = neuron.backward(upstream_grad)

        st.success("✅ 梯度计算完成")

        with st.expander("📋 梯度信息", expanded=True):
            grad_info = [
                {
                    "参数": "W_Q",
                    "形状": f"{neuron.W_Q.shape}",
                    "梯度范数": f'{np.linalg.norm(gradients["grad_W_Q"]):.6f}',
                },
                {
                    "参数": "W_K",
                    "形状": f"{neuron.W_K.shape}",
                    "梯度范数": f'{np.linalg.norm(gradients["grad_W_K"]):.6f}',
                },
                {
                    "参数": "W_V",
                    "形状": f"{neuron.W_V.shape}",
                    "梯度范数": f'{np.linalg.norm(gradients["grad_W_V"]):.6f}',
                },
            ]
            grad_df = pd.DataFrame(grad_info)
            st.markdown(grad_df.to_markdown(index=False))

        # 参数更新
        st.markdown("---")
        st.subheader("📊 3. 参数更新")

        old_W_Q = neuron.W_Q.copy()
        old_W_K = neuron.W_K.copy()
        old_W_V = neuron.W_V.copy()

        neuron.update_parameters(learning_rate)

        st.success("✅ 参数已更新")

        change_Q = np.linalg.norm(neuron.W_Q - old_W_Q)
        change_K = np.linalg.norm(neuron.W_K - old_W_K)
        change_V = np.linalg.norm(neuron.W_V - old_W_V)

        changes_data = [
            {"参数": "W_Q", "变化量": f"{change_Q:.6f}"},
            {"参数": "W_K", "变化量": f"{change_K:.6f}"},
            {"参数": "W_V", "变化量": f"{change_V:.6f}"},
        ]
        changes_df = pd.DataFrame(changes_data)
        st.markdown(changes_df.to_markdown(index=False))

        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.subheader("🔬 数值稳定性诊断")

        st.info("💡 Attention特有问题：注意力权重分布、Softmax溢出、Query-Key相似度")

        stability_issues = []

        # 1. 检查梯度
        grad_Q_check = StabilityChecker.check_gradient(
            gradients["grad_W_Q"].flatten(), "W_Q梯度"
        )
        stability_issues.append(grad_Q_check)

        grad_K_check = StabilityChecker.check_gradient(
            gradients["grad_W_K"].flatten(), "W_K梯度"
        )
        stability_issues.append(grad_K_check)

        grad_V_check = StabilityChecker.check_gradient(
            gradients["grad_W_V"].flatten(), "W_V梯度"
        )
        stability_issues.append(grad_V_check)

        # 2. 检查注意力权重分布（关键！）
        attn_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
        max_attn_weight = np.max(attention_weights)

        if max_attn_weight > 0.9:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "注意力权重过度集中",
                    "value": f"{max_attn_weight:.4f}",
                    "threshold": "> 0.9",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大权重": f"{max_attn_weight:.4f}",
                        "熵值": f"{attn_entropy:.4f}",
                        "权重分布": ", ".join([f"{w:.3f}" for w in attention_weights]),
                    },
                    "solution": [
                        "检查Query和Key的初始化",
                        "使用temperature缩放",
                        "添加attention dropout",
                        "检查输入是否过于相似",
                    ],
                    "explanation": "注意力权重过度集中在某个位置，可能导致信息瓶颈",
                }
            )
        elif attn_entropy < 0.5:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "注意力熵值过低",
                    "value": f"{attn_entropy:.4f}",
                    "threshold": "< 0.5",
                    "icon": "🟡",
                    "severity": "low",
                    "details": {
                        "熵值": f"{attn_entropy:.4f}",
                        "最大权重": f"{max_attn_weight:.4f}",
                        "权重分布": ", ".join([f"{w:.3f}" for w in attention_weights]),
                    },
                    "solution": [
                        "可能是正常现象（某个位置确实最重要）",
                        "如果总是如此，检查模型设计",
                        "考虑使用多头注意力",
                    ],
                    "explanation": "注意力分布的信息熵较低，关注较为集中",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "注意力权重分布",
                    "value": f"熵={attn_entropy:.4f}",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "熵值": f"{attn_entropy:.4f}",
                        "最大权重": f"{max_attn_weight:.4f}",
                        "权重和": f"{np.sum(attention_weights):.6f}",
                    },
                }
            )

        # 3. 检查attention scores (softmax前)
        scores = neuron.forward_history["scores"]
        score_max = np.max(np.abs(scores))

        if score_max > 10:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "Attention Score过大",
                    "value": f"{score_max:.2f}",
                    "threshold": "> 10",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大score": f"{score_max:.2f}",
                        "Score范围": f"[{np.min(scores):.2f}, {np.max(scores):.2f}]",
                        "缩放因子": f"√{d_model} = {np.sqrt(d_model):.2f}",
                    },
                    "solution": [
                        "检查Query和Key的范数",
                        "增大缩放因子（√d_k）",
                        "使用LayerNorm",
                        "检查输入是否已归一化",
                    ],
                    "explanation": "Score过大会导致softmax饱和，梯度消失",
                }
            )

        # 4. 检查Q, K, V的范数
        Q_norm = np.linalg.norm(neuron.forward_history["Q"])
        K_norm = np.linalg.norm(neuron.forward_history["K"])
        V_norm = np.linalg.norm(neuron.forward_history["V"])

        if Q_norm > 10 or K_norm > 10 or V_norm > 10:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "Q/K/V范数过大",
                    "value": f"max={max(Q_norm, K_norm, V_norm):.2f}",
                    "threshold": "> 10",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "Q范数": f"{Q_norm:.4f}",
                        "K范数": f"{K_norm:.4f}",
                        "V范数": f"{V_norm:.4f}",
                    },
                    "solution": [
                        "使用Xavier初始化",
                        "添加LayerNorm在投影后",
                        "检查输入数据范围",
                    ],
                    "explanation": "Q/K/V范数过大会导致attention score爆炸",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "Q/K/V范数",
                    "value": f"Q={Q_norm:.2f}, K={K_norm:.2f}, V={V_norm:.2f}",
                    "icon": "🟢",
                    "severity": "none",
                }
            )

        # 5. 检查输出
        output_check = StabilityChecker.check_activation(output, "Attention输出")
        stability_issues.append(output_check)

        # 6. 检查学习率
        combined_param_norm = np.sqrt(
            np.linalg.norm(neuron.W_Q) ** 2
            + np.linalg.norm(neuron.W_K) ** 2
            + np.linalg.norm(neuron.W_V) ** 2
        )
        combined_grad_norm = np.sqrt(
            np.linalg.norm(gradients["grad_W_Q"]) ** 2
            + np.linalg.norm(gradients["grad_W_K"]) ** 2
            + np.linalg.norm(gradients["grad_W_V"]) ** 2
        )
        lr_check = StabilityChecker.check_learning_rate(
            learning_rate, combined_grad_norm, combined_param_norm
        )
        stability_issues.append(lr_check)

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 Attention神经元稳定性诊断"
        )

        st.markdown("---")
        st.info(
            f"""
        💡 **注意力机制特性与稳定性**：
        
        **动态权重**: 注意力权重根据输入动态计算，不是固定参数
        - 当前分布熵: {attn_entropy:.4f}
        - 最大权重: {max_attn_weight:.4f}
        
        **选择性关注**: 自动学习关注重要的位置
        - Q范数: {Q_norm:.4f}
        - K范数: {K_norm:.4f}
        - V范数: {V_norm:.4f}
        
        **可解释性**: 注意力权重可以可视化，了解模型在"看"什么
        - 权重和: {np.sum(attention_weights):.6f} (应该=1.0)
        
        **无距离限制**: 可以关注任意位置，不受距离影响
        - Score范围: [{np.min(scores):.2f}, {np.max(scores):.2f}]
        - 缩放因子: √{d_model} = {np.sqrt(d_model):.2f}
        
        **应用**: Transformer、BERT、GPT等现代NLP模型的核心
        
        **总参数量**: {total_params} 个（不包括动态计算的注意力权重）
        
        **关键洞察**: Query问"我想找什么"，Key回答"我是什么"，相似度高的Key对应的Value会被更多关注！
        
        **Scaled Dot-Product Attention**:
        - Score = Q·K^T / √d_k
        - 缩放的目的：防止点积过大导致softmax饱和
        - 当d_k很大时，点积值方差为d_k，缩放后方差为1
        """
        )


def single_neuron_tab(CHINESE_SUPPORTED=True):
    """
    单神经元可视化标签页 - 主UI界面
    """

    if CHINESE_SUPPORTED:
        st.header("🧬 单神经元：理解神经网络的基本单元")

        st.markdown(
            """
        ### 💡 核心思想
        
        神经元是神经网络的**最小计算单元**。理解单个神经元如何工作，就能理解整个神经网络的运作原理。
        
        **选择不同类型的神经元，实际体验它们的工作方式！**
        """
        )

        st.markdown("---")

        # 神经元类型选择
        st.subheader("🎯 选择神经元类型")

        neuron_type = st.selectbox(
            "选择要探索的神经元类型",
            [
                "全连接神经元 (Dense/FC)",
                "卷积神经元 (Conv)",
                "循环神经元 (RNN)",
                "GRU神经元",
                "LSTM神经元",
                "注意力机制 (Attention)",
            ],
            help="不同类型的神经元适用于不同的任务",
        )

        st.markdown("---")

        # 根据神经元类型显示不同的配置和演示
        if neuron_type == "全连接神经元 (Dense/FC)":
            render_dense_neuron()
        elif neuron_type == "卷积神经元 (Conv)":
            render_conv_neuron()
        elif neuron_type == "循环神经元 (RNN)":
            render_rnn_neuron()
        elif neuron_type == "GRU神经元":
            render_gru_neuron()
        elif neuron_type == "LSTM神经元":
            render_lstm_neuron()
        elif neuron_type == "注意力机制 (Attention)":
            render_attention_neuron()

        # 教学说明
        st.markdown("---")
        with st.expander("📚 详细说明：神经元如何工作", expanded=False):
            st.markdown(
                """
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
            """
            )

        # 不同类型神经网络的神经元对比
        st.markdown("---")
        with st.expander("🔬 扩展：不同类型神经网络的神经元工作原理", expanded=False):
            st.markdown(
                """
            ### 🎯 核心问题：其他类型的神经网络神经元也是这样工作的吗？
            
            **答案：基本原理相同，但具体实现有所不同！**
            
            所有神经网络都遵循相同的三步流程：**前向传播 → 反向传播 → 参数更新**，但不同类型的神经网络在"如何计算"上有所差异。
            
            ---
            
            ### 1️⃣ **全连接神经元（Fully Connected / Dense）** - 本页演示
            
            **特点**：每个输入都与每个输出相连
            
            **前向传播**：
            $$
            y = \\text{activation}(w^T x + b)
            $$
            
            **参数**：
            - 权重矩阵 $W$：大小为 $\\text{output\\_size} \\times \\text{input\\_size}$
            - 偏置向量 $b$：大小为 $\\text{output\\_size}$
            
            **优点**：表达能力强，可以学习任意函数
            
            **缺点**：参数量大，容易过拟合，不适合处理图像等高维数据
            
            **应用**：MLP、全连接层
            
            ---
            
            ### 2️⃣ **卷积神经元（Convolutional Neuron）** 
            
            **特点**：局部连接 + 权值共享
            
            **前向传播**：
            $$
            y[i,j] = \\text{activation}\\left(\\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} w[m,n] \\cdot x[i+m, j+n] + b\\right)
            $$
            
            **关键差异**：
            - 🔍 **局部感受野**：每个神经元只关注输入的一小块区域（如 3×3 或 5×5）
            - 🔄 **权值共享**：同一个卷积核在整个输入上滑动，参数被复用
            - 📦 **参数量**：卷积核大小 $k \\times k \\times C_{in}$，与输入大小无关！
            
            **例子**：
            - 全连接：28×28 图像 → 100 神经元 = 78,400 参数
            - 卷积：3×3 卷积核 → 100 个 = 900 参数（减少 87 倍！）
            
            **反向传播**：使用卷积的转置操作（转置卷积）
            
            **优点**：参数少、平移不变性、适合图像
            
            **应用**：CNN、ResNet、VGG
            
            ---
            
            ### 3️⃣ **循环神经元（Recurrent Neuron）**
            
            **特点**：有记忆，处理序列数据
            
            **前向传播**：
            $$
            h_t = \\text{activation}(W_{hh} h_{t-1} + W_{xh} x_t + b)
            $$
            
            **关键差异**：
            - 🔁 **时间循环**：当前状态 $h_t$ 依赖于前一时刻状态 $h_{t-1}$
            - 📝 **记忆机制**：隐藏状态 $h_t$ 存储历史信息
            - ⏱️ **时间展开**：在时间维度上展开成多个时间步
            
            **参数**：
            - $W_{xh}$：输入到隐藏的权重
            - $W_{hh}$：隐藏到隐藏的权重（记忆权重）
            - $W_{hy}$：隐藏到输出的权重
            
            **反向传播**：BPTT（Backpropagation Through Time）- 沿时间反向传播
            
            **问题**：梯度消失/爆炸（长期依赖难以学习）
            
            **应用**：RNN、语言模型、时序预测
            
            ---
            
            ### 4️⃣ **LSTM 神经元（Long Short-Term Memory）**
            
            **特点**：改进的 RNN，解决长期依赖问题
            
            **前向传播**（复杂得多！）：
            $$
            \\begin{aligned}
            f_t &= \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) && \\text{（遗忘门）} \\\\
            i_t &= \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) && \\text{（输入门）} \\\\
            \\tilde{C}_t &= \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C) && \\text{（候选记忆）} \\\\
            C_t &= f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t && \\text{（更新记忆）} \\\\
            o_t &= \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) && \\text{（输出门）} \\\\
            h_t &= o_t \\odot \\tanh(C_t) && \\text{（输出）}
            \\end{aligned}
            $$
            
            **关键差异**：
            - 🚪 **门控机制**：3 个门（遗忘门、输入门、输出门）控制信息流动
            - 💾 **细胞状态** $C_t$：长期记忆，可以跨越很多时间步
            - 🎛️ **精细控制**：决定保留什么、忘记什么、输出什么
            
            **参数数量**：4 倍于普通 RNN（因为有 4 组权重）
            
            **反向传播**：梯度通过细胞状态直接流动，缓解梯度消失
            
            **优点**：能学习长期依赖
            
            **应用**：机器翻译、语音识别、文本生成
            
            ---
            
            ### 5️⃣ **注意力机制神经元（Attention Neuron）**
            
            **特点**：动态加权，关注重要信息
            
            **前向传播**：
            $$
            \\begin{aligned}
            \\text{score}(h_i, s) &= h_i^T W s && \\text{（相似度计算）} \\\\
            \\alpha_i &= \\frac{\\exp(\\text{score}(h_i, s))}{\\sum_j \\exp(\\text{score}(h_j, s))} && \\text{（注意力权重）} \\\\
            c &= \\sum_i \\alpha_i h_i && \\text{（加权求和）}
            \\end{aligned}
            $$
            
            **关键差异**：
            - 🎯 **动态权重**：注意力权重 $\\alpha_i$ 根据输入动态计算（不是固定参数）
            - 🔍 **选择性关注**：自动学习关注哪些部分重要
            - 🌐 **全局视野**：可以关注任意位置，没有距离限制
            
            **反向传播**：梯度同时流向查询、键、值
            
            **优点**：捕获长距离依赖、可解释性强
            
            **应用**：Transformer、BERT、GPT
            
            ---
            
            ### 6️⃣ **Transformer 自注意力神经元（Self-Attention）**
            
            **特点**：并行处理，无序列限制
            
            **前向传播**：
            $$
            \\begin{aligned}
            Q &= XW_Q, \\quad K = XW_K, \\quad V = XW_V \\\\
            \\text{Attention}(Q,K,V) &= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
            \\end{aligned}
            $$
            
            **关键差异**：
            - 🔀 **完全并行**：所有位置同时计算，不像 RNN 必须按顺序
            - 🎭 **多头注意力**：多个注意力头关注不同方面
            - 📊 **位置编码**：需要额外添加位置信息
            
            **参数**：$W_Q, W_K, W_V$ 三个投影矩阵
            
            **反向传播**：梯度可以直接流动，没有梯度消失问题
            
            **优点**：并行高效、长距离依赖、性能最强
            
            **应用**：大语言模型（GPT、BERT、LLaMA）
            
            ---
            
            ### 📊 **总结对比表**
            
            | 神经元类型 | 连接方式 | 参数共享 | 适用场景 | 主要优势 |
            |:----------|:---------|:---------|:---------|:---------|
            | **全连接** | 全连接 | 否 | 表格数据、MLP | 表达能力强 |
            | **卷积** | 局部连接 | 是（卷积核） | 图像、空间数据 | 参数少、平移不变 |
            | **RNN** | 循环连接 | 是（时间维度） | 序列、时序数据 | 处理变长序列 |
            | **LSTM** | 循环+门控 | 是（时间维度） | 长序列 | 长期依赖 |
            | **注意力** | 动态权重 | 否（权重动态） | 序列到序列 | 选择性关注 |
            | **Transformer** | 自注意力 | 否 | NLP、多模态 | 并行、长距离 |
            
            ---
            
            ### 🎓 **关键洞察**
            
            1. **统一框架**：所有神经元都遵循"前向 → 反向 → 更新"的范式
            
            2. **设计哲学**：
               - 全连接：**通用性** - 能学任何东西，但代价高
               - 卷积：**归纳偏置** - 利用局部性和平移不变性
               - RNN/LSTM：**时序归纳偏置** - 利用时间依赖性
               - Transformer：**最小归纳偏置** - 让数据说话，需要大量数据
            
            3. **计算复杂度**：
               - 全连接：$O(n^2)$ - 输入输出大小的乘积
               - 卷积：$O(k^2 \\cdot C \\cdot H \\cdot W)$ - 卷积核大小 × 通道 × 输出大小
               - RNN：$O(T \\cdot n^2)$ - 时间步 × 状态大小平方
               - Transformer：$O(T^2 \\cdot d)$ - 序列长度平方 × 特征维度
            
            4. **梯度流动**：
               - 全连接/卷积：直接反向传播
               - RNN：BPTT，容易梯度消失
               - LSTM：细胞状态缓解梯度消失
               - Transformer：跳跃连接 + LayerNorm 保持梯度
            
            5. **从简单到复杂**：
               - 理解全连接神经元 → 理解 MLP
               - 理解卷积神经元 → 理解 CNN
               - 理解循环神经元 → 理解 RNN/LSTM
               - 理解注意力神经元 → 理解 Transformer
            
            **本页展示的是最基础的全连接神经元，它是所有其他神经元的基础！**
            
            ---
            
            ### 💡 **实践建议**
            
            - 🎯 **从简单开始**：先掌握全连接神经元（本页）
            - 📚 **逐步深入**：理解每种神经元的设计动机
            - 🧪 **动手实验**：在其他标签页中探索 CNN、RNN 等
            - 🔬 **对比学习**：对比不同神经元在同一任务上的表现
            - 📖 **阅读论文**：了解每种架构的原始论文和发展历程
            
            **记住：神经元的核心是"加权求和 + 非线性激活 + 梯度优化"，万变不离其宗！**
            """
            )

    else:
        st.header("🧬 Single Neuron: Understanding the Basic Unit")
        st.info("English version - Coming soon!")


if __name__ == "__main__":
    # 测试代码
    neuron = SingleNeuron(input_size=3, activation="relu")
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
