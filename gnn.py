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
        feature_norm = np.linalg.norm(H)
        feature_check = StabilityChecker.check_activation(H.flatten(), "输入节点特征")
        stability_issues.append(feature_check)

        # 2. 检查聚合后的特征
        aggregated_check = StabilityChecker.check_activation(
            messages.flatten(), "聚合后特征"
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
            stability_issues.append(
                {
                    "status": "error",
                    "type": "严重过平滑",
                    "value": f"{avg_similarity:.4f}",
                    "threshold": "> 0.95",
                    "icon": "🔴",
                    "severity": "critical",
                    "details": {
                        "平均相似度": f"{avg_similarity:.4f}",
                        "最大相似度": f"{max_similarity:.4f}",
                        "节点数": num_nodes,
                        "特征维度": feature_dim,
                    },
                    "solution": [
                        "减少GNN层数",
                        "使用残差连接（如ResGCN）",
                        "使用PairNorm/GraphNorm",
                        "使用Jumping Knowledge Networks",
                        "添加自环（self-loops）权重",
                    ],
                    "explanation": "所有节点特征高度相似，失去了节点间的区分度，这是深层GNN的典型问题",
                }
            )
        elif avg_similarity > 0.85:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "轻度过平滑",
                    "value": f"{avg_similarity:.4f}",
                    "threshold": "> 0.85",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "平均相似度": f"{avg_similarity:.4f}",
                        "最大相似度": f"{max_similarity:.4f}",
                        "节点数": num_nodes,
                    },
                    "solution": [
                        "监控更深层的相似度变化",
                        "考虑添加残差连接",
                        "使用节点自适应聚合",
                    ],
                    "explanation": "节点特征相似度较高，继续加深可能导致过平滑",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "节点特征区分度",
                    "value": f"平均相似度={avg_similarity:.4f}",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "平均相似度": f"{avg_similarity:.4f}",
                        "最大相似度": f"{max_similarity:.4f}",
                        "节点数": num_nodes,
                    },
                }
            )

        # 5. 邻接矩阵谱分析
        eigenvalues = np.linalg.eigvals(A_hat)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        if max_eigenvalue > 1.1:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "邻接矩阵特征值过大",
                    "value": f"{max_eigenvalue:.4f}",
                    "threshold": "> 1.1",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大特征值": f"{max_eigenvalue:.4f}",
                        "归一化方法": "对称归一化",
                        "理想范围": "[0, 1]",
                    },
                    "solution": [
                        "检查归一化是否正确",
                        "使用谱归一化",
                        "添加自环权重",
                        "使用GCN的归一化技巧",
                    ],
                    "explanation": "特征值>1可能导致特征爆炸，影响训练稳定性",
                }
            )
        else:
            stability_issues.append(
                {
                    "status": "success",
                    "type": "邻接矩阵特征值",
                    "value": f"{max_eigenvalue:.4f}",
                    "icon": "🟢",
                    "severity": "none",
                    "details": {
                        "最大特征值": f"{max_eigenvalue:.4f}",
                        "特征值范围": f"[{np.min(np.abs(eigenvalues)):.4f}, {max_eigenvalue:.4f}]",
                    },
                }
            )

        # 6. 度分布检查
        degree_sum = np.sum(A, axis=1)
        max_degree = np.max(degree_sum)
        min_degree = np.min(degree_sum)
        degree_variance = np.var(degree_sum)

        if max_degree / (min_degree + 1) > 10:
            stability_issues.append(
                {
                    "status": "warning",
                    "type": "度分布不平衡",
                    "value": f"最大/最小={max_degree/(min_degree+1):.1f}",
                    "threshold": "> 10",
                    "icon": "🟡",
                    "severity": "medium",
                    "details": {
                        "最大度": f"{max_degree:.0f}",
                        "最小度": f"{min_degree:.0f}",
                        "平均度": f"{np.mean(degree_sum):.2f}",
                        "方差": f"{degree_variance:.2f}",
                    },
                    "solution": [
                        "使用度归一化（GCN标准）",
                        "使用注意力机制（GAT）",
                        "对高度节点进行采样",
                        "使用GraphSAINT等采样方法",
                    ],
                    "explanation": "度分布不平衡会导致高度节点特征主导，低度节点信息不足",
                }
            )

        # 显示诊断结果
        StabilityChecker.display_issues(
            stability_issues, title="🔬 GNN数值稳定性诊断报告"
        )

        st.markdown("---")
        st.info(
            f"""
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
        """
        )

    # ==========================================
    # 新增：图结构问题诊断（第4个核心问题）
    # ==========================================
    st.markdown("---")
    st.markdown("### ⚠️ 图结构问题诊断")

    st.info(
        """
    💡 **这一部分回答："什么时候会出问题？"**
    
    自动检测当前图结构和GNN配置的潜在问题。
    """
    )

    # 自动检测机制
    issues = []
    warnings = []

    # 计算图的统计信息
    degrees = np.sum(A > 0, axis=1)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    isolated_nodes = np.sum(degrees == 0)

    # 检测1: 孤立节点
    if isolated_nodes > 0:
        issues.append(
            {
                "问题": "孤立节点",
                "检测结果": f"{isolated_nodes}个节点没有任何连接（度=0）",
                "影响": "这些节点无法从邻居获取信息，特征无法更新",
                "解决方案": "添加自环、连接到虚拟节点、或删除孤立节点",
            }
        )

    # 检测2: 图过于稀疏
    edge_density = np.sum(A > 0) / (num_nodes * num_nodes)
    if edge_density < 0.05 and num_nodes > 10:
        warnings.append(
            {
                "问题": "图过于稀疏",
                "检测结果": f"边密度={edge_density:.3f}，平均度={avg_degree:.1f}",
                "影响": "信息传播慢，需要更多层才能到达远距离节点",
                "解决方案": "增加边、使用跳跃连接、或考虑图增强",
            }
        )

    # 检测3: 图过于稠密
    if edge_density > 0.5 and num_nodes > 10:
        warnings.append(
            {
                "问题": "图过于稠密",
                "检测结果": f"边密度={edge_density:.3f}，接近全连接",
                "影响": "计算开销大，过平滑风险高",
                "解决方案": "采样邻居、使用注意力机制、或稀疏化图",
            }
        )

    # 检测4: 度分布不均
    degree_std = np.std(degrees)
    if degree_std > avg_degree:
        warnings.append(
            {
                "问题": "度分布不均",
                "检测结果": f"度标准差({degree_std:.1f}) > 平均度({avg_degree:.1f})",
                "影响": "Hub节点主导信息流，尾部节点信息不足",
                "解决方案": "度归一化、注意力加权、或邻居采样",
            }
        )

    # 检测5: 层数过多导致过平滑
    if num_layers > 3:
        warnings.append(
            {
                "问题": "层数过多（过平滑风险）",
                "检测结果": f"{num_layers}层GNN",
                "影响": "节点特征趋于相同，丧失区分度",
                "解决方案": "减少层数(2-3层)、残差连接、或Jumping Knowledge",
            }
        )

    # 显示检测结果
    if issues:
        st.error(f"🚨 检测到 {len(issues)} 个严重问题！")

        for idx, issue in enumerate(issues, 1):
            with st.expander(f"❌ 问题 {idx}: {issue['问题']}", expanded=True):
                st.markdown(f"**检测结果**: {issue['检测结果']}")
                st.markdown(f"**影响**: {issue['影响']}")
                st.markdown(f"**✅ 解决方案**: {issue['解决方案']}")

    if warnings:
        st.warning(f"⚠️ 检测到 {len(warnings)} 个潜在问题")

        for idx, warning in enumerate(warnings, 1):
            with st.expander(f"⚠️ 警告 {idx}: {warning['问题']}", expanded=False):
                st.markdown(f"**检测结果**: {warning['检测结果']}")
                st.markdown(f"**影响**: {warning['影响']}")
                st.markdown(f"**💡 建议**: {warning['解决方案']}")

    if not issues and not warnings:
        st.success("✅ 当前图结构和配置无明显问题！")

    # 图统计信息
    st.markdown("---")
    st.markdown("### 📊 图结构统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("节点数", num_nodes)
        st.metric("边数", int(np.sum(A > 0)))

    with col2:
        st.metric("平均度", f"{avg_degree:.2f}")
        st.metric("边密度", f"{edge_density:.3f}")

    with col3:
        st.metric("最大度", int(max_degree))
        st.metric("最小度", int(min_degree))

    with col4:
        st.metric("孤立节点", int(isolated_nodes))
        st.metric("GNN层数", num_layers)

    # 常见问题诊断表格
    st.markdown("---")
    st.markdown("### 📋 GNN常见问题速查表")

    diagnostic_table = pd.DataFrame(
        {
            "问题症状": [
                "所有节点预测相同",
                "训练时间很长",
                "远距离节点信息传不到",
                "Hub节点效果好，尾部节点差",
                "增加层数反而效果变差",
                "显存不足",
            ],
            "可能原因": [
                "过平滑（层数过多）",
                "图过于稠密",
                "图稀疏且层数少",
                "度分布不均",
                "过平滑问题",
                "邻接矩阵过大或batch过大",
            ],
            "诊断方法": [
                "计算特征方差，检查是否趋近0",
                "计算边密度和邻居数",
                "计算图直径和层数",
                "可视化度分布直方图",
                "对比不同层数的效果",
                "监控GPU显存占用",
            ],
            "解决方案": [
                "减少层数、残差连接、JK-Net",
                "邻居采样、稀疏化图",
                "增加层数或添加虚拟边",
                "度归一化、注意力机制",
                "固定2-3层、使用GAT或ResGNN",
                "邻居采样、小batch、梯度累积",
            ],
        }
    )

    st.dataframe(diagnostic_table, use_container_width=True, height=280)

    # 配置建议
    st.markdown("---")
    st.markdown("### 💡 GNN配置最佳实践")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **👍 推荐配置**：
        
        - **层数**: 2-3层（避免过平滑）
        - **归一化**: 对称归一化 $\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}$
        - **自环**: 添加自环保留自身信息
        - **激活函数**: ReLU或LeakyReLU
        - **Dropout**: 在消息传递后使用
        - **残差连接**: 深层GNN必备
        """
        )

    with col2:
        st.markdown(
            """
        **👎 应避免的配置**：
        
        - ❌ 层数>5（除非用ResGCN）
        - ❌ 无归一化的邻接矩阵
        - ❌ 忽略孤立节点
        - ❌ 全连接图（计算爆炸）
        - ❌ 无自环（丢失自身特征）
        - ❌ 过大的batch（显存不足）
        """
        )

    # 不同场景的建议
    st.markdown("---")
    st.markdown("### 🎯 不同场景的GNN选择")

    scenario_table = pd.DataFrame(
        {
            "图类型": ["社交网络", "分子图", "知识图谱", "推荐系统", "交通网络"],
            "特点": [
                "大规模、稀疏、Hub节点",
                "小图、密集连接",
                "异构、多关系",
                "二部图、极度稀疏",
                "网格结构、规则",
            ],
            "推荐模型": [
                "GraphSAGE(采样)",
                "GCN或GIN",
                "R-GCN或HGT",
                "LightGCN或PinSage",
                "GAT或Spatial GNN",
            ],
            "建议层数": ["2-3层", "3-5层", "2-3层", "2层", "2-4层"],
            "关键技巧": ["邻居采样", "边特征", "关系建模", "负采样", "位置编码"],
        }
    )

    st.dataframe(scenario_table, use_container_width=True)

    st.info(
        """
    💡 **GNN调试技巧**：
    
    1. **先检查图结构**：孤立节点、度分布、连通性
    2. **从浅层开始**：先2层，效果不好再加层
    3. **可视化特征**：每层后检查特征是否趋同
    4. **对比基线**：与MLP（无图结构）对比，确认图信息有用
    5. **逐步优化**：归一化 → 自环 → 残差 → 注意力
    """
    )


if __name__ == "__main__":
    # 独立运行时的测试
    gnn_tab(True)
