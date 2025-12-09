"""
GNN图神经网络数学原理模块

v2.2.0 新增：
- 数值稳定性自动检测
- 过平滑(over-smoothing)检测
- 节点特征范数检测
- 邻接矩阵谱分析
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from simple_latex import display_latex
from utils.numerical_stability_checker import StabilityChecker

from utils.visualization import ChartBuilder
from utils.exceptions import ComputationError


def gnn_tab(CHINESE_SUPPORTED):
    """GNN标签页内容"""

    # 定义默认参数
    # 使用动态示例生成器
    from utils.example_generator import get_dynamic_example

    try:
        example = get_dynamic_example("gnn")
        num_nodes = example["num_nodes"]
        feature_dim = example["feature_dim"]
    except Exception as e:
        # 如果动态生成失败，使用默认值
        num_nodes = 8
        feature_dim = 3

    # 使用动态参数建议器
    from utils.parameter_suggester import get_suggested_params

    try:
        suggested_params = get_suggested_params(
            "gnn",
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            task_complexity="medium",
        )
        num_layers = suggested_params["num_layers"]
        hidden_dims = suggested_params["hidden_dims"]
        dropout = suggested_params["dropout"]
        learning_rate = suggested_params["learning_rate"]
    except Exception as e:
        # 如果动态建议失败，使用默认值
        num_layers = 2
        hidden_dims = [feature_dim * 4, feature_dim * 8]
        dropout = 0.5
        learning_rate = 0.001

    st.header("🕸️ GNN图神经网络数学原理")

    # 初始化图表工具
    chart_builder = ChartBuilder()

    display_latex("H^{(l+1)} = \\sigma(\\tilde{A} H^{(l)} W^{(l)})")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 符号说明")
        st.markdown("- Ã: 归一化的邻接矩阵")
        st.markdown("- $H_l$: 第l层的节点特征矩阵")
        st.markdown("- $W_l$: 第l层的权重矩阵")
        st.markdown("- $\\sigma$: 激活函数")

        # 创建示例图
        G = nx.erdos_renyi_graph(num_nodes, 0.4, seed=42)
        pos = nx.spring_layout(G, seed=42)

        # 随机分配节点标签用于可视化
        node_labels = np.random.randint(0, 4, num_nodes)

        # 可视化图结构
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
        node_color = []
        colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_color.append(colors[node_labels[node] % len(colors)])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=20,
                color=node_color,
                line_width=2,
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=(
                        "图结构与真实标签"
                        if CHINESE_SUPPORTED
                        else "Graph Structure and True Labels"
                    ),
                    font=dict(size=16),
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="#888", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
            ),
        )
        st.plotly_chart(fig, width="stretch")

        # 邻接矩阵
        A = nx.adjacency_matrix(G).todense()
        st.markdown("### 邻接矩阵 A")
        st.dataframe(
            pd.DataFrame(
                A,
                index=[f"Node {i}" for i in range(num_nodes)],
                columns=[f"Node {i}" for i in range(num_nodes)],
            )
        )

    with col2:
        st.markdown("### 归一化邻接矩阵计算")

        # 添加自环
        A_tilde = A + np.eye(num_nodes)
        st.markdown("#### 步骤1: 添加自环 Ã = A + I")
        st.dataframe(
            pd.DataFrame(
                A_tilde,
                index=[f"Node {i}" for i in range(num_nodes)],
                columns=[f"Node {i}" for i in range(num_nodes)],
            )
        )

        # 计算度矩阵
        D_tilde = np.diag(np.sum(A_tilde, axis=1))
        st.markdown("#### 步骤2: 度矩阵 D̃")
        st.dataframe(
            pd.DataFrame(
                D_tilde,
                index=[f"Node {i}" for i in range(num_nodes)],
                columns=[f"Node {i}" for i in range(num_nodes)],
            )
        )

        # 归一化
        try:
            D_tilde_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
            A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        except np.linalg.LinAlgError as e:
            # 处理奇异矩阵情况
            try:
                D_tilde_sqrt = np.sqrt(D_tilde)
                D_tilde_inv_sqrt = np.zeros_like(D_tilde_sqrt)
                non_zero_mask = D_tilde_sqrt > 1e-10
                D_tilde_inv_sqrt[non_zero_mask] = 1.0 / D_tilde_sqrt[non_zero_mask]
                A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
            except Exception as calc_error:
                raise ComputationError(
                    operation="图拉普拉斯矩阵归一化",
                    error_details=f"奇异矩阵处理失败: {str(calc_error)}",
                ) from e

        st.markdown(
            "#### 步骤3: 对称归一化 $ \\tilde{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} $"
        )
        st.dataframe(
            pd.DataFrame(
                A_hat.round(3),
                index=[f"Node {i}" for i in range(num_nodes)],
                columns=[f"Node {i}" for i in range(num_nodes)],
            )
        )

    # 消息传递可视化
    st.markdown("---")
    st.markdown("### 🔗 消息传递机制")

    # 使用动态示例生成器
    try:
        example = get_dynamic_example("gnn")
        H = example["node_features"]
        feature_dim = example["feature_dim"]
    except Exception as e:
        # 如果动态生成失败，使用默认值
        feature_dim = 3
        H = np.random.randn(num_nodes, feature_dim).round(2)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📊 节点特征数据")
        st.markdown("**每个节点都有特征值，就像每个人都有不同的特点**")

        # 可视化特征矩阵
        fig = px.imshow(
            H,
            labels=dict(x="特征维度", y="节点", color="特征值"),
            color_continuous_scale="RdYlBu_r",
            title="节点特征热力图",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

        # 显示具体数值（可选）
        show_details = st.checkbox("🔍 显示具体数值")
        if show_details:
            st.dataframe(
                pd.DataFrame(
                    H.round(2),
                    index=[f"节点{i}" for i in range(num_nodes)],
                    columns=[f"特征{j}" for j in range(feature_dim)],
                )
            )

        # 权重矩阵
        st.markdown("#### ⚙️ 连接权重")
        st.markdown("**权重决定了节点间信息传递的强度**")

        W = np.random.randn(feature_dim, feature_dim).round(2)
        fig = px.imshow(
            W,
            labels=dict(x="输出特征", y="输入特征", color="权重值"),
            color_continuous_scale="RdBu",
            title="权重矩阵热力图",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### 🔄 信息传递过程")

        # 计算消息传递
        messages = A_hat @ H
        st.markdown("**第1步：邻居信息聚合**")
        st.markdown("*每个节点收集邻居的信息，就像和朋友聊天一样*")

        # 可视化消息聚合
        fig = px.imshow(
            messages.round(3),
            labels=dict(x="特征", y="节点", color="聚合值"),
            color_continuous_scale="Viridis",
            title="邻居信息聚合结果",
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, width="stretch")

        # 线性变换
        transformed = messages @ W
        st.markdown("**第2步：信息变换**")
        st.markdown("*通过权重矩阵重新组合信息，就像重新整理思路*")

        fig = px.imshow(
            transformed.round(3),
            labels=dict(x="输出特征", y="节点", color="变换值"),
            color_continuous_scale="Plasma",
            title="信息变换结果",
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, width="stretch")

        # 激活函数
        activated = F.relu(torch.tensor(transformed)).numpy()
        st.markdown("**第3步：激活处理**")
        st.markdown("*ReLU就像一个过滤器：保留有用的信息，去掉负值*")

        # 对比激活前后
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(range(len(activated.flatten()))),
                y=transformed.flatten(),
                name="激活前",
                marker_color="lightblue",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Bar(
                x=list(range(len(activated.flatten()))),
                y=activated.flatten(),
                name="激活后",
                marker_color="orange",
                opacity=0.7,
            )
        )
        fig.update_layout(
            title="ReLU激活效果对比",
            xaxis_title="特征索引",
            yaxis_title="数值",
            height=300,
            barmode="overlay",
        )
        st.plotly_chart(fig, width="stretch")

        # 激活函数可视化
        st.markdown("**🎯 激活函数工作原理**")
        x_vals = np.linspace(-5, 5, 100)
        relu_vals = np.maximum(0, x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=relu_vals, mode="lines", name="ReLU"))
        fig.update_layout(
            title="ReLU函数：f(x) = max(0, x)",
            xaxis_title="输入值",
            yaxis_title="输出值",
            height=250,
        )
        st.plotly_chart(fig, width="stretch")

        # 显示数值（可选）
        if show_details:
            st.markdown("**最终输出数值**")
            st.dataframe(
                pd.DataFrame(
                    activated.round(3),
                    index=[f"节点{i}" for i in range(num_nodes)],
                    columns=[f"输出{j}" for j in range(feature_dim)],
                )
            )
        
        # ==================== 数值稳定性检测 ====================
        st.markdown("---")
        st.markdown("### 🔬 GNN数值稳定性诊断")
        
        st.info("💡 GNN特有问题：过平滑(over-smoothing)、梯度消失、节点特征退化")
        
        stability_issues = []
        
        # 1. 检查节点特征范数
        feature_norm = np.linalg.norm(features)
        feature_check = StabilityChecker.check_activation(
            features.flatten(), "输入节点特征"
        )
        stability_issues.append(feature_check)
        
        # 2. 检查聚合后的特征
        aggregated_check = StabilityChecker.check_activation(
            aggregated.flatten(), "聚合后特征"
        )
        stability_issues.append(aggregated_check)
        
        # 3. 检查输出特征
        output_check = StabilityChecker.check_activation(
            activated.flatten(), "GNN输出特征"
        )
        stability_issues.append(output_check)
        
        # 4. 过平滑检测（关键！）
        # 计算节点特征之间的余弦相似度
        feature_norms = np.linalg.norm(activated, axis=1, keepdims=True)
        normalized_features = activated / (feature_norms + 1e-8)
        similarity_matrix = np.dot(normalized_features, normalized_features.T)
        
        # 排除对角线
        off_diagonal_mask = ~np.eye(num_nodes, dtype=bool)
        avg_similarity = np.mean(similarity_matrix[off_diagonal_mask])
        max_similarity = np.max(similarity_matrix[off_diagonal_mask])
        
        if avg_similarity > 0.95:
            stability_issues.append({
                'status': 'error',
                'type': '严重过平滑',
                'value': f'{avg_similarity:.4f}',
                'threshold': '> 0.95',
                'icon': '🔴',
                'severity': 'critical',
                'details': {
                    '平均相似度': f'{avg_similarity:.4f}',
                    '最大相似度': f'{max_similarity:.4f}',
                    '节点数': num_nodes,
                    '特征维度': feature_dim
                },
                'solution': [
                    '减少GNN层数',
                    '使用残差连接（如ResGCN）',
                    '使用PairNorm/GraphNorm',
                    '使用Jumping Knowledge Networks',
                    '添加自环（self-loops）权重'
                ],
                'explanation': '所有节点特征高度相似，失去了节点间的区分度，这是深层GNN的典型问题'
            })
        elif avg_similarity > 0.85:
            stability_issues.append({
                'status': 'warning',
                'type': '轻度过平滑',
                'value': f'{avg_similarity:.4f}',
                'threshold': '> 0.85',
                'icon': '🟡',
                'severity': 'medium',
                'details': {
                    '平均相似度': f'{avg_similarity:.4f}',
                    '最大相似度': f'{max_similarity:.4f}',
                    '节点数': num_nodes
                },
                'solution': [
                    '监控更深层的相似度变化',
                    '考虑添加残差连接',
                    '使用节点自适应聚合'
                ],
                'explanation': '节点特征相似度较高，继续加深可能导致过平滑'
            })
        else:
            stability_issues.append({
                'status': 'success',
                'type': '节点特征区分度',
                'value': f'平均相似度={avg_similarity:.4f}',
                'icon': '🟢',
                'severity': 'none',
                'details': {
                    '平均相似度': f'{avg_similarity:.4f}',
                    '最大相似度': f'{max_similarity:.4f}',
                    '节点数': num_nodes
                }
            })
        
        # 5. 邻接矩阵谱分析
        eigenvalues = np.linalg.eigvals(normalized_adj)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        if max_eigenvalue > 1.1:
            stability_issues.append({
                'status': 'warning',
                'type': '邻接矩阵特征值过大',
                'value': f'{max_eigenvalue:.4f}',
                'threshold': '> 1.1',
                'icon': '🟡',
                'severity': 'medium',
                'details': {
                    '最大特征值': f'{max_eigenvalue:.4f}',
                    '归一化方法': '对称归一化',
                    '理想范围': '[0, 1]'
                },
                'solution': [
                    '检查归一化是否正确',
                    '使用谱归一化',
                    '添加自环权重',
                    '使用GCN的归一化技巧'
                ],
                'explanation': '特征值>1可能导致特征爆炸，影响训练稳定性'
            })
        else:
            stability_issues.append({
                'status': 'success',
                'type': '邻接矩阵特征值',
                'value': f'{max_eigenvalue:.4f}',
                'icon': '🟢',
                'severity': 'none',
                'details': {
                    '最大特征值': f'{max_eigenvalue:.4f}',
                    '特征值范围': f'[{np.min(np.abs(eigenvalues)):.4f}, {max_eigenvalue:.4f}]'
                }
            })
        
        # 6. 度分布检查
        degree_sum = np.sum(adj_matrix, axis=1)
        max_degree = np.max(degree_sum)
        min_degree = np.min(degree_sum)
        degree_variance = np.var(degree_sum)
        
        if max_degree / (min_degree + 1) > 10:
            stability_issues.append({
                'status': 'warning',
                'type': '度分布不平衡',
                'value': f'最大/最小={max_degree/(min_degree+1):.1f}',
                'threshold': '> 10',
                'icon': '🟡',
                'severity': 'medium',
                'details': {
                    '最大度': f'{max_degree:.0f}',
                    '最小度': f'{min_degree:.0f}',
                    '平均度': f'{np.mean(degree_sum):.2f}',
                    '方差': f'{degree_variance:.2f}'
                },
                'solution': [
                    '使用度归一化（GCN标准）',
                    '使用注意力机制（GAT）',
                    '对高度节点进行采样',
                    '使用GraphSAINT等采样方法'
                ],
                'explanation': '度分布不平衡会导致高度节点特征主导，低度节点信息不足'
            })
        
        # 显示诊断结果
        StabilityChecker.display_issues(stability_issues, 
                                       title="🔬 GNN数值稳定性诊断报告")
        
        st.markdown("---")
        st.info(f"""
        💡 **GNN健康指标总结**：
        
        **节点特征**：
        - 输入范数: {feature_norm:.4f}
        - 输出范围: [{np.min(activated):.2f}, {np.max(activated):.2f}]
        
        **过平滑指标**：
        - 平均节点相似度: {avg_similarity:.4f} (建议<0.85)
        - 最大节点相似度: {max_similarity:.4f}
        
        **图结构**：
        - 邻接矩阵最大特征值: {max_eigenvalue:.4f} (建议≤1.0)
        - 度分布: 最小{min_degree:.0f}, 最大{max_degree:.0f}, 平均{np.mean(degree_sum):.2f}
        
        **典型GNN问题**：
        1. **过平滑(Over-smoothing)**: 深层GNN导致所有节点特征趋同
           - 症状：节点相似度>0.9
           - 解决：残差连接、PairNorm、减少层数
        
        2. **梯度消失**: 类似于深层神经网络
           - 症状：梯度范数<1e-7
           - 解决：残差连接、LayerNorm、控制层数
        
        3. **度不平衡**: Hub节点主导信息流
           - 症状：度分布方差大
           - 解决：度归一化、注意力机制、采样
        
        4. **特征退化**: 所有节点特征收敛到相同值
           - 症状：特征方差趋近于0
           - 解决：Jumping Knowledge、混合不同层的特征
        
        **推荐实践**：
        - GCN: 通常2-3层最优
        - GAT: 可以到4-5层（注意力缓解过平滑）
        - ResGCN: 可以到10+层（残差连接）
        """)


if __name__ == "__main__":
    # 独立运行时的测试
    gnn_tab(True)
