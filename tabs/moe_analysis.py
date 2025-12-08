"""
MoE (Mixture of Experts) 计算解剖标签页
Mixture of Experts Computational Analysis Tab

深入解剖混合专家模型的数值计算过程
核心理念：让你看到MoE每一步到底算了什么数值，为什么这样计算
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from utils.memory_analyzer import get_tensor_memory


class MoELayer(nn.Module):
    """简化的MoE层实现"""

    def __init__(self, input_dim, output_dim, num_experts, expert_capacity_factor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor

        # 路由网络 (Gating)
        self.gate = nn.Linear(input_dim, num_experts)

        # 专家网络
        self.experts = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_experts)]
        )

        # 专家容量
        self.expert_capacity = int(expert_capacity_factor * input_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # 路由决策
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 选择top-k专家 (这里简化为top-1)
        top_k_probs, selected_experts = torch.topk(gate_probs, k=1, dim=-1)

        # 专家处理
        outputs = torch.zeros(batch_size, self.output_dim)

        for expert_idx in range(self.num_experts):
            # 找到使用当前专家的样本
            mask = (selected_experts == expert_idx).squeeze(-1)
            if mask.sum() > 0:
                expert_input = x[mask]
                expert_output = self.experts[expert_idx](expert_input)
                outputs[mask] = expert_output

        return outputs, {
            "gate_probs": gate_probs,
            "selected_experts": selected_experts,
            "expert_usage": self._calculate_expert_usage(selected_experts),
        }

    def _calculate_expert_usage(self, selected_experts):
        """计算专家使用率"""
        usage = torch.zeros(self.num_experts)
        for i in range(self.num_experts):
            usage[i] = (selected_experts == i).sum().item()
        return usage


def explain_moe_computation():
    """解释MoE的数值计算过程"""
    st.markdown(
        """
    ### 🧠 MoE 数值计算过程详解
    
    #### 核心计算步骤
    
    **第1步：路由网络计算**
    ```
    输入: x ∈ ℝ^(batch_size, input_dim)
    
    # 线性变换
    logits = x · W_gate + b_gate
    # 其中: W_gate ∈ ℝ^(input_dim, num_experts)
    
    # Softmax归一化
    gate_probs = softmax(logits) 
    # gate_probs[i,j] = exp(logits[i,j]) / Σ_k exp(logits[i,k])
    # 含义: 第i个样本选择第j个专家的概率
    
    数值例子:
    x = [0.5, -0.3]
    W_gate = [[0.1, 0.2], [-0.1, 0.3]]
    logits = x · W_gate = [0.43, -0.15]
    gate_probs = softmax([0.43, -0.15]) = [0.64, 0.36]
    ```
    
    **关键问题：为什么选择这个专家？**
    - 路由网络学习输入特征与专家能力的匹配
    - 数值大小反映专家对该输入的适合程度
    
    **第2步：专家选择与计算**
    ```
    # Top-k选择 (k=1)
    selected_expert = argmax(gate_probs, dim=1)
    top_k_probs = topk(gate_probs, k=1)
    
    # 专家网络前向传播
    for each selected expert j:
        expert_output_j = x · W_expert_j + b_expert_j
        # W_expert_j ∈ ℝ^(input_dim, output_dim)
    
    数值例子:
    selected_expert = 0 (选择专家0)
    top_k_probs = [0.64]
    
    W_expert_0 = [[0.3, 0.1], [0.2, -0.4]]
    expert_output_0 = x · W_expert_0 = [0.49, 0.33]
    ```
    
    **第3步：加权组合输出**
    ```
    # 加权求和
    final_output = Σ (gate_probs[i,j] × expert_output_j)
    
    数值例子:
    final_output = 0.64 × [0.49, 0.33] = [0.3136, 0.2112]
    ```
    
    **第4步：梯度反向传播**
    ```
    # 对专家输出的梯度
    ∂L/∂expert_output_j = ∂L/∂final_output × gate_probs[i,j]
    
    # 对路由网络的梯度  
    ∂L/∂gate_probs[i,j] = expert_output_j · ∂L/∂final_output
    
    # 对专家权重的梯度
    ∂L/∂W_expert_j = x^T · (∂L/∂expert_output_j)
    
    关键洞察：只有被选中的专家会获得梯度更新！
    ```
    
    #### 数值稳定性问题
    
    **1. Softmax饱和**
    - 问题：logits值过大导致梯度消失
    - 现象：gate_probs接近[1,0,0,...]，专家选择过于确定
    - 解决：温度缩放、梯度裁剪
    
    **2. 专家权重爆炸**
    - 问题：W_expert_j值过大导致输出溢出
    - 现象：expert_output_j包含NaN或Inf
    - 解决：权重初始化、批归一化
    
    **3. 梯度消失**
    - 问题：gate_probs过小导致专家梯度消失
    - 现象：某些专家永远得不到更新
    - 解决：负载均衡损失、专家容量限制
    """
    )


def plot_expert_usage(expert_usage, expert_names=None):
    """绘制专家使用率"""
    if expert_names is None:
        expert_names = [f"专家{i+1}" for i in range(len(expert_usage))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=expert_names,
                y=expert_usage,
                text=[f"{usage:.1%}" for usage in expert_usage],
                textposition="auto",
                marker_color="lightblue",
            )
        ]
    )

    fig.update_layout(
        title="专家使用率分布", xaxis_title="专家", yaxis_title="使用次数", height=400
    )

    return fig


def visualize_moe_architecture(num_experts, expert_usage=None):
    """可视化MoE架构"""
    if not expert_usage is None:
        expert_usage = torch.ones(num_experts)

    fig = go.Figure()

    # 输入节点
    fig.add_trace(
        go.Scatter(
            x=[1],
            y=[3],
            mode="markers+text",
            marker=dict(size=30, color="lightblue", line=dict(color="black", width=2)),
            text="输入",
            textposition="middle center",
            showlegend=False,
            name="输入",
        )
    )

    # 路由器
    fig.add_trace(
        go.Scatter(
            x=[2.5],
            y=[3],
            mode="markers+text",
            marker=dict(size=25, color="yellow", line=dict(color="black", width=2)),
            text="路由器",
            textposition="middle center",
            showlegend=False,
            name="路由器",
        )
    )

    # 专家网络
    expert_colors = [
        "lightgreen",
        "lightcoral",
        "lightpink",
        "lightyellow",
        "lightgray",
    ]
    for i in range(num_experts):
        y_pos = 4 + i * 0.8
        color = expert_colors[i % len(expert_colors)]

        # 根据使用率调整透明度
        if expert_usage is not None:
            opacity = 0.3 + 0.7 * (expert_usage[i] / max(expert_usage))
        else:
            opacity = 0.8

        fig.add_trace(
            go.Scatter(
                x=[4],
                y=[y_pos],
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(color="black", width=2),
                    opacity=opacity,
                ),
                text=f"专家{i+1}",
                textposition="middle center",
                showlegend=False,
                name=f"专家{i+1}",
            )
        )

        # 连接线：路由器到专家
        fig.add_trace(
            go.Scatter(
                x=[2.5, 4],
                y=[3, y_pos],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 输出节点
    fig.add_trace(
        go.Scatter(
            x=[5.5],
            y=[3],
            mode="markers+text",
            marker=dict(size=30, color="lightblue", line=dict(color="black", width=2)),
            text="输出",
            textposition="middle center",
            showlegend=False,
            name="输出",
        )
    )

    # 连接线：专家到输出
    for i in range(num_experts):
        y_pos = 4 + i * 0.8
        fig.add_trace(
            go.Scatter(
                x=[4, 5.5],
                y=[y_pos, 3],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 连接线：输入到路由器
    fig.add_trace(
        go.Scatter(
            x=[1, 2.5],
            y=[3, 3],
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="MoE (Mixture of Experts) 架构示意图",
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 6.5]),
        yaxis=dict(visible=False, range=[2, 4 + num_experts * 0.8]),
        height=300 + num_experts * 40,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def moe_analysis_tab(chinese_supported=True):
    """MoE计算解剖主函数"""

    st.header("🧠 MoE (Mixture of Experts) 计算解剖台")
    st.markdown(
        """
    > **核心理念**：深入解剖MoE每一步的数值计算过程
    
    **关键问题**：
    - 路由网络到底算了什么概率？
    - 专家网络的输出如何加权组合？
    - 梯度如何在专家间反向传播？
    - 为什么某些数值会导致专家选择偏差？
    """
    )

    st.markdown("---")

    # MoE计算过程解析
    with st.expander("🧠 MoE数值计算过程（点击展开）", expanded=False):
        explain_moe_computation()

    st.markdown("---")

    # 分析模式选择
    st.subheader("🔧 选择计算解剖模式")

    analysis_mode = st.radio(
        "计算解剖模式",
        ["路由网络计算过程", "专家网络数值计算", "梯度反向传播解剖", "数值稳定性分析"],
        horizontal=True,
    )

    if analysis_mode == "路由网络计算过程":
        st.markdown("---")
        st.subheader("🧮 路由网络计算过程解剖")

        st.markdown(
            """
        **核心问题**：路由网络到底是怎么计算专家选择概率的？
        """
        )

        # 可配置参数
        col1, col2 = st.columns(2)

        with col1:
            input_dim = st.number_input("输入维度", 2, 4, 2, key="route_input_dim")
            num_experts = st.number_input("专家数量", 2, 4, 2, key="route_experts")

        with col2:
            st.markdown("**示例输入**")
            example_input = st.text_input("输入向量", "0.5, -0.3", key="example_input")

        if st.button("🔍 开始计算解剖", type="primary"):
            # 解析输入
            x = torch.tensor([float(val.strip()) for val in example_input.split(",")])

            # 创建路由网络权重
            with st.spinner("计算中..."):
                # 手动设置权重矩阵用于展示
                W_gate = torch.tensor([[0.1, 0.2], [-0.1, 0.3]])
                b_gate = torch.tensor([0.0, 0.0])

                # 第1步：线性变换
                logits = torch.matmul(x, W_gate) + b_gate
                st.markdown("#### 📊 第1步：线性变换计算")
                st.code(
                    f"""
输入向量 x = {x.tolist()}
权重矩阵 W_gate = {W_gate.tolist()}
偏置 b_gate = {b_gate.tolist()}

计算过程：
logits = x · W_gate + b_gate
       = {x[0]:.2f} × {W_gate[0,0]:.2f} + {x[1]:.2f} × {W_gate[1,0]:.2f} + {b_gate[0]:.2f}, {x[0]:.2f} × {W_gate[0,1]:.2f} + {x[1]:.2f} × {W_gate[1,1]:.2f} + {b_gate[1]:.2f}
       = {logits[0]:.3f}, {logits[1]:.3f}
                """
                )

                # 第2步：Softmax计算
                exp_logits = torch.exp(logits)
                sum_exp = torch.sum(exp_logits)
                gate_probs = exp_logits / sum_exp

                st.markdown("#### 📊 第2步：Softmax归一化计算")
                st.code(
                    f"""
线性变换结果 logits = {logits.tolist()}

计算过程：
exp(logits) = [exp({logits[0]:.3f}), exp({logits[1]:.3f})]
             = [{exp_logits[0]:.3f}, {exp_logits[1]:.3f}]

sum(exp) = {sum_exp:.3f}

gate_probs = exp(logits) / sum(exp)
          = [{exp_logits[0]:.3f}/{sum_exp:.3f}, {exp_logits[1]:.3f}/{sum_exp:.3f}]
          = [{gate_probs[0]:.3f}, {gate_probs[1]:.3f}]

含义：
- 专家0被选择的概率：{gate_probs[0]:.1%}
- 专家1被选择的概率：{gate_probs[1]:.1%}
                """
                )

                # 第3步：Top-K选择
                top_k_probs, selected_experts = torch.topk(gate_probs, k=1)

                st.markdown("#### 📊 第3步：Top-K选择计算")
                st.code(
                    f"""
路由概率 gate_probs = {gate_probs.tolist()}

Top-1选择：
- 选择专家：{selected_experts[0].item()} (概率最高的专家)
- 选择概率：{top_k_probs[0]:.3f}
- 未选择专家的概率被丢弃

关键洞察：
- 只有概率最高的专家会被激活
- 其他专家的计算会被跳过，节省计算资源
                """
                )

                # 第4步：数值敏感性分析
                st.markdown("#### 📊 数值敏感性分析")

                # 测试不同输入的影响
                test_inputs = [
                    torch.tensor([1.0, 1.0]),
                    torch.tensor([-1.0, -1.0]),
                    torch.tensor([2.0, -2.0]),
                    torch.tensor([0.1, 0.1]),
                ]

                sensitivity_data = []
                for test_x in test_inputs:
                    test_logits = torch.matmul(test_x, W_gate) + b_gate
                    test_gate_probs = F.softmax(test_logits, dim=0)
                    sensitivity_data.append(
                        {
                            "输入": test_x.tolist(),
                            "logits": test_logits.tolist(),
                            "概率": test_gate_probs.tolist(),
                            "熵": -torch.sum(
                                test_gate_probs * torch.log(test_gate_probs)
                            ).item(),
                        }
                    )

                st.markdown("**不同输入的数值变化**：")
                df = pd.DataFrame(sensitivity_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown(
                    """
                **数值洞察**：
                - 输入越大，logits值越大，概率分布越极端
                - 输入越小，概率分布越均匀
                - 熵值反映选择的不确定性：熵越大越不确定
                """
                )

    elif analysis_mode == "专家网络数值计算":
        st.markdown("---")
        st.subheader("🔬 专家网络数值计算解剖")

        st.markdown(
            """
        **核心问题**：被选中的专家具体是怎么计算输出的？
        """
        )

        # 配置专家网络
        col1, col2 = st.columns(2)

        with col1:
            input_dim = st.number_input("输入维度", 2, 4, 2, key="expert_input_dim")
            output_dim = st.number_input("输出维度", 2, 4, 2, key="expert_output_dim")

        with col2:
            st.markdown("**专家权重矩阵**")
            expert_weights = st.text_area(
                "专家权重 (每行一个专家)", "0.3, 0.1\n0.2, -0.4", key="expert_weights"
            )

        if st.button("🔬 开始专家计算解剖", type="primary"):
            with st.spinner("计算中..."):
                # 解析权重
                weights_lines = expert_weights.strip().split("\n")
                expert_weights_list = []
                for line in weights_lines:
                    weights = [float(val.strip()) for val in line.split(",")]
                    expert_weights_list.append(weights)

                # 示例输入
                x = torch.tensor([0.5, -0.3])

                st.markdown("#### 📊 专家网络计算过程")

                for i, W_expert in enumerate(expert_weights_list):
                    # 检查权重数量是否匹配
                    expected_size = input_dim * output_dim
                    if len(W_expert) != expected_size:
                        st.error(
                            f"专家{i}权重数量不匹配！期望{expected_size}个，实际{len(W_expert)}个"
                        )
                        continue

                    W_expert_tensor = torch.tensor(W_expert).view(input_dim, output_dim)

                    # 计算专家输出
                    expert_output = torch.matmul(x, W_expert_tensor)

                    st.markdown(f"**专家 {i} 计算过程**：")
                    st.code(
                        f"""
专家 {i} 权重矩阵：
{W_expert_tensor.tolist()}

输入向量：{x.tolist()}

计算过程：
output = x · W_expert
       = {x[0]:.2f} × {W_expert_tensor[0,0]:.2f} + {x[1]:.2f} × {W_expert_tensor[1,0]:.2f}, {x[0]:.2f} × {W_expert_tensor[0,1]:.2f} + {x[1]:.2f} × {W_expert_tensor[1,1]:.2f}
       = {expert_output[0]:.3f}, {expert_output[1]:.3f}

数值分析：
- 输出范围：[{expert_output.min():.3f}, {expert_output.max():.3f}]
- 输出模长：{torch.norm(expert_output):.3f}
- 输出方向：{expert_output / torch.norm(expert_output)}
                    """
                    )

                    # 可视化
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=[f"输出{i}" for i in range(output_dim)],
                            y=expert_output.tolist(),
                            name=f"专家{i}",
                        )
                    )
                    fig.update_layout(title=f"专家{i}输出分布", yaxis_title="输出值")
                    st.plotly_chart(fig, use_container_width=True)

                # 专家对比分析
                st.markdown("#### 📊 专家对比分析")

                all_outputs = []
                for i, W_expert in enumerate(expert_weights_list):
                    # 检查权重数量
                    expected_size = input_dim * output_dim
                    if len(W_expert) != expected_size:
                        continue

                    W_expert_tensor = torch.tensor(W_expert).view(input_dim, output_dim)
                    expert_output = torch.matmul(x, W_expert_tensor)
                    all_outputs.append(expert_output.tolist())

                comparison_df = pd.DataFrame(
                    {
                        "专家": [f"专家{i}" for i in range(len(all_outputs))],
                        "输出1": [out[0] for out in all_outputs],
                        "输出2": [out[1] for out in all_outputs],
                        "模长": [
                            torch.norm(torch.tensor(out)).item() for out in all_outputs
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                st.markdown(
                    """
                **专家差异分析**：
                - 不同专家对相同输入产生不同输出
                - 模长反映专家的"激活强度"
                - 路由网络根据任务需求选择最合适的专家
                """
                )

    elif analysis_mode == "梯度反向传播解剖":
        st.markdown("---")
        st.subheader("🌊 梯度反向传播解剖")

        st.markdown(
            """
        **核心问题**：MoE的梯度到底是怎么反向传播的？
        """
        )

        # 简化的MoE梯度计算示例
        st.markdown("#### 📊 MoE梯度计算步骤")

        st.code(
            """
示例：单个样本的MoE梯度计算

前向传播：
x = [0.5, -0.3]  # 输入
gate_probs = [0.7, 0.3]  # 路由概率
expert_outputs = [[0.49, 0.33], [0.21, -0.15]]  # 专家输出
final_output = 0.7 × [0.49, 0.33] + 0.3 × [0.21, -0.15] = [0.394, 0.186]

反向传播：
∂L/∂final_output = [0.1, -0.2]  # 假设的损失梯度

1. 对专家输出的梯度：
∂L/∂expert_0 = ∂L/∂final_output × gate_probs[0] = [0.1, -0.2] × 0.7 = [0.07, -0.14]
∂L/∂expert_1 = ∂L/∂final_output × gate_probs[1] = [0.1, -0.2] × 0.3 = [0.03, -0.06]

2. 对专家权重的梯度：
∂L/∂W_expert_0 = x^T × ∂L/∂expert_0 = [[0.5], [-0.3]] × [[0.07, -0.14]]
                     = [[0.5×0.07, 0.5×(-0.14)], [-0.3×0.07, -0.3×(-0.14)]]
                     = [[0.035, -0.07], [-0.021, 0.042]]

3. 对路由网络的梯度：
∂L/∂gate_probs[0] = expert_0 · ∂L/∂final_output = [0.49, 0.33] · [0.1, -0.2] = 0.49×0.1 + 0.33×(-0.2) = -0.017
∂L/∂gate_probs[1] = expert_1 · ∂L/∂final_output = [0.21, -0.15] · [0.1, -0.2] = 0.21×0.1 + (-0.15)×(-0.2) = 0.051

关键洞察：
- 只有被选中的专家获得梯度更新
- 路由概率作为权重调节专家梯度
- 未被选中的专家权重不会更新
        """
        )

        # 梯度可视化
        st.markdown("#### 📊 梯度数值可视化")

        # 创建梯度数据
        expert_grads = [[0.035, -0.07], [-0.021, 0.042]]
        gate_grads = [-0.017, 0.051]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("专家梯度", "路由梯度"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        fig.add_trace(
            go.Bar(
                x=["W[0,0]", "W[0,1]", "W[1,0]", "W[1,1]"],
                y=[
                    expert_grads[0][0],
                    expert_grads[0][1],
                    expert_grads[1][0],
                    expert_grads[1][1],
                ],
                name="专家0",
                marker_color="blue",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=["W[0,0]", "W[0,1]", "W[1,0]", "W[1,1]"],
                y=[
                    expert_grads[1][0],
                    expert_grads[1][1],
                    expert_grads[0][0],
                    expert_grads[0][1],
                ],
                name="专家1",
                marker_color="red",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=["专家0", "专家1"], y=gate_grads, marker_color="green"),
            row=1,
            col=2,
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        **梯度分析**：
        - 专家0的梯度更大，因为路由概率更高
        - 负梯度表示需要减小权重值
        - 正梯度表示需要增大权重值
        """
        )

    else:  # 数值稳定性分析
        st.markdown("---")
        st.subheader("⚠️ 数值稳定性分析")

        st.markdown(
            """
        **核心问题**：MoE在什么情况下会出现数值问题？
        """
        )

        st.markdown("#### 📊 常见数值问题")

        # 1. Softmax饱和
        st.markdown("**1. Softmax饱和问题**")
        st.code(
            """
问题：logits值过大导致梯度消失

正常情况：
logits = [0.5, -0.3]
gate_probs = softmax([0.5, -0.3]) = [0.64, 0.36]
梯度正常传播

饱和情况：
logits = [10.0, -10.0]
gate_probs = softmax([10.0, -10.0]) ≈ [1.0, 0.0]
梯度几乎为零（数值下溢）

解决方案：
- 温度缩放：logits / temperature
- 梯度裁剪：限制梯度最大值
- 权重初始化：避免极端值
        """
        )

        # 2. 专家权重爆炸
        st.markdown("**2. 专家权重爆炸**")
        st.code(
            """
问题：专家权重过大导致输出溢出

正常权重：
W = [[0.3, 0.1], [0.2, -0.4]]
output = x · W = [0.23, -0.14]

爆炸权重：
W = [[100.0, 50.0], [80.0, -120.0]]
output = x · W = [65.0, 71.0]  # 可能导致后续计算溢出

检测方法：
- 监控权重范数：||W|| > threshold
- 检查激活值范围：|output| > threshold
- 观察梯度范数：||∂L/∂W|| > threshold
        """
        )

        # 3. 负载不均衡
        st.markdown("**3. 负载不均衡问题**")
        st.code(
            """
问题：某些专家很少被使用，导致训练不均衡

负载均衡指标：
load_balance = 1.0 - std(expert_usage_rates)

理想情况：
expert_usage_rates = [0.5, 0.5]  # 完全均衡
load_balance = 1.0 - 0.0 = 1.0

不均衡情况：
expert_usage_rates = [0.9, 0.1]  # 严重不均衡  
load_balance = 1.0 - 0.4 = 0.6

解决方案：
- 添加负载均衡损失：L_balance = α × Var(usage_rates)
- 专家容量限制：max_samples_per_expert
- 噪声专家：随机选择专家增加探索
        """
        )

        # 数值稳定性检测工具
        st.markdown("#### 🔧 数值稳定性检测工具")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**检测阈值设置**")
            temp_threshold = st.slider("温度阈值", 0.1, 10.0, 1.0)
            weight_threshold = st.slider("权重范数阈值", 1.0, 100.0, 10.0)
            grad_threshold = st.slider("梯度范数阈值", 0.1, 10.0, 1.0)

        with col2:
            st.markdown("**诊断结果**")
            st.metric("Softmax温度", "正常", "0.8 < 1.0")
            st.metric("权重范数", "警告", "15.2 > 10.0")
            st.metric("梯度范数", "正常", "0.5 < 1.0")
            st.metric("负载均衡", "良好", "0.85 > 0.7")

        st.markdown(
            """
        **诊断建议**：
        - ✅ 正常：继续训练
        - ⚠️ 警告：监控数值，准备调整
        - ❌ 异常：立即停止，调整参数
        """
        )

    # 总结
    st.markdown("---")
    st.subheader("💡 MoE计算解剖核心要点")

    st.markdown(
        """
    ### 🎯 关键计算洞察
    
    **路由网络的数值本质**：
    - 线性变换 + Softmax = 概率分布
    - 数值大小反映专家适合度
    - 温度参数控制分布尖锐度
    
    **专家选择的数值影响**：
    - Top-K策略：只激活概率最高的专家
    - 负载均衡：确保专家使用均匀
    - 容量限制：防止专家过载
    
    **梯度传播的数值特点**：
    - 选择性传播：只有被选中的专家获得梯度
    - 概率加权：路由概率调节梯度大小
    - 稀疏性：大部分专家权重不更新
    
    ### ⚠️ 数值稳定性关键
    
    **Softmax稳定性**：
    - logits值范围：[-10, 10] 较为安全
    - 温度参数：0.1-2.0 适中范围
    - 梯度裁剪：防止梯度爆炸
    
    **权重初始化**：
    - 专家权重：Xavier/He初始化
    - 路由权重：小随机初始化
    - 偏置项：通常初始化为0
    
    **训练监控**：
    - 专家使用率：避免专家坍塌
    - 梯度范数：检测数值异常
    - 激活值范围：防止数值溢出
    """
    )


if __name__ == "__main__":
    # 测试运行
    moe_analysis_tab()
