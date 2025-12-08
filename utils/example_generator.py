"""
动态示例生成器
Dynamic Example Generator

基于用户当前选择的参数生成演示示例
替代硬编码的示例数据
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st


class ExampleGenerator:
    """动态示例生成器

    基于用户当前选择的参数生成相应的演示示例
    """

    @staticmethod
    def get_current_user_params() -> Dict[str, Any]:
        """获取用户当前选择的参数"""
        user_params = {}

        # 从session state获取参数
        if hasattr(st, "session_state"):
            # CNN相关参数
            if "cnn_input_channels" in st.session_state:
                user_params["input_channels"] = st.session_state.cnn_input_channels
            if "cnn_output_channels" in st.session_state:
                user_params["output_channels"] = st.session_state.cnn_output_channels
            if "cnn_kernel_size" in st.session_state:
                user_params["kernel_size"] = st.session_state.cnn_kernel_size
            if "cnn_stride" in st.session_state:
                user_params["stride"] = st.session_state.cnn_stride
            if "cnn_padding" in st.session_state:
                user_params["padding"] = st.session_state.cnn_padding
            if "cnn_input_h" in st.session_state:
                user_params["input_height"] = st.session_state.cnn_input_h
            if "cnn_input_w" in st.session_state:
                user_params["input_width"] = st.session_state.cnn_input_w

            # ViT相关参数
            if "vit_img_size" in st.session_state:
                user_params["img_size"] = st.session_state.vit_img_size
            if "vit_patch_size" in st.session_state:
                user_params["patch_size"] = st.session_state.vit_patch_size
            if "vit_d_model" in st.session_state:
                user_params["d_model"] = st.session_state.vit_d_model
            if "vit_num_heads" in st.session_state:
                user_params["num_heads"] = st.session_state.vit_num_heads

            # GNN相关参数
            if "gnn_num_nodes" in st.session_state:
                user_params["num_nodes"] = st.session_state.gnn_num_nodes
            if "gnn_feature_dim" in st.session_state:
                user_params["feature_dim"] = st.session_state.gnn_feature_dim

            # LSTM相关参数
            if "lstm_hidden" in st.session_state:
                user_params["hidden_size"] = st.session_state.lstm_hidden
            if "lstm_layers" in st.session_state:
                user_params["num_layers"] = st.session_state.lstm_layers

        return user_params

    @staticmethod
    def generate_cnn_example() -> Dict[str, Any]:
        """生成CNN示例"""
        user_params = ExampleGenerator.get_current_user_params()

        # 使用用户参数或默认值
        input_size = user_params.get("input_height", 32)
        kernel_size = user_params.get("kernel_size", 3)
        stride = user_params.get("stride", 1)
        padding = user_params.get("padding", 1)

        # 计算输出尺寸
        output_size = (input_size + 2 * padding - kernel_size) // stride + 1

        # 生成示例数据
        input_matrix = np.random.randn(input_size, input_size).round(2)
        kernel = np.random.randn(kernel_size, kernel_size).round(2)

        return {
            "input_size": input_size,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "output_size": output_size,
            "input_matrix": input_matrix,
            "kernel": kernel,
            "calculation_formula": f"H_out = floor(({input_size} + 2*{padding} - {kernel_size}) / {stride}) + 1 = {output_size}",
        }

    @staticmethod
    def generate_vit_example() -> Dict[str, Any]:
        """生成ViT示例"""
        user_params = ExampleGenerator.get_current_user_params()

        # 使用用户参数或默认值
        img_size = user_params.get("img_size", 224)
        patch_size = user_params.get("patch_size", 16)
        d_model = user_params.get("d_model", 768)
        num_heads = user_params.get("num_heads", 12)

        # 计算patch数量
        num_patches = (img_size // patch_size) ** 2
        seq_len = num_patches + 1  # +1 for CLS token

        # 计算注意力矩阵大小
        attn_matrix_size = seq_len**2

        return {
            "img_size": img_size,
            "patch_size": patch_size,
            "num_patches": num_patches,
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "attn_matrix_size": attn_matrix_size,
            "calculation": {
                "patches": f"{img_size}÷{patch_size} = {img_size//patch_size}, 共{(img_size//patch_size)}²={num_patches}个patches",
                "embedding": f"每个patch: [3, {patch_size}, {patch_size}] = {3*patch_size*patch_size}维向量",
                "attention": f"Attention矩阵: {seq_len}×{seq_len} = {attn_matrix_size:,}个元素",
                "heads": f"{num_heads}个头: {num_heads}×{attn_matrix_size:,} = {num_heads*attn_matrix_size:,}个元素",
            },
        }

    @staticmethod
    def generate_gnn_example() -> Dict[str, Any]:
        """生成GNN示例"""
        user_params = ExampleGenerator.get_current_user_params()

        # 使用用户参数或默认值
        num_nodes = user_params.get("num_nodes", 8)
        feature_dim = user_params.get("feature_dim", 3)

        # 生成示例数据
        # 创建随机邻接矩阵（确保连通）
        adj_matrix = np.random.randint(0, 2, (num_nodes, num_nodes))
        adj_matrix = (adj_matrix + adj_matrix.T) // 2  # 对称化
        np.fill_diagonal(adj_matrix, 1)  # 自环

        # 生成节点特征
        node_features = np.random.randn(num_nodes, feature_dim).round(2)

        # 生成权重矩阵
        weight_matrix = np.random.randn(feature_dim, feature_dim).round(2)

        return {
            "num_nodes": num_nodes,
            "feature_dim": feature_dim,
            "adj_matrix": adj_matrix,
            "node_features": node_features,
            "weight_matrix": weight_matrix,
            "calculation": {
                "message_passing": f"每个节点聚合来自{num_nodes-1}个邻居的信息",
                "feature_transform": f"特征维度从{feature_dim}变换到{feature_dim}",
                "computation": f"计算复杂度: O({num_nodes} × {feature_dim}²)",
            },
        }

    @staticmethod
    def generate_math_example() -> Dict[str, Any]:
        """生成数学推导示例"""
        user_params = ExampleGenerator.get_current_user_params()

        # 使用用户参数或默认值
        num_nodes = user_params.get("num_nodes", 8)

        # 生成示例图
        # 创建随机但连通的图
        adj_matrix = np.random.randint(0, 2, (num_nodes, num_nodes))
        adj_matrix = (adj_matrix + adj_matrix.T) // 2
        np.fill_diagonal(adj_matrix, 0)  # 无自环

        # 确保连通（添加一个环）
        for i in range(num_nodes):
            adj_matrix[i, (i + 1) % num_nodes] = 1
            adj_matrix[(i + 1) % num_nodes, i] = 1

        # 生成度矩阵
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

        # 计算拉普拉斯矩阵
        laplacian = degree_matrix - adj_matrix

        return {
            "num_nodes": num_nodes,
            "adj_matrix": adj_matrix,
            "degree_matrix": degree_matrix,
            "laplacian": laplacian,
            "formulas": {
                "degree": f"D[i,i] = Σ_j A[i,j]",
                "laplacian": f"L = D - A",
                "example": f"对于{num_nodes}个节点的图",
            },
        }

    @staticmethod
    def generate_lstm_example() -> Dict[str, Any]:
        """生成LSTM示例"""
        user_params = ExampleGenerator.get_current_user_params()

        # 使用用户参数或默认值
        hidden_size = user_params.get("hidden_size", 256)
        num_layers = user_params.get("num_layers", 2)
        sequence_length = user_params.get("sequence_length", 20)

        # 计算参数量
        num_gates = 4  # LSTM有4个门
        first_layer_params = num_gates * (
            sequence_length * hidden_size + hidden_size * hidden_size
        )
        other_layer_params = num_gates * (
            hidden_size * hidden_size + hidden_size * hidden_size
        )
        total_params = first_layer_params + other_layer_params * (num_layers - 1)

        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "sequence_length": sequence_length,
            "num_gates": num_gates,
            "first_layer_params": first_layer_params,
            "other_layer_params": other_layer_params,
            "total_params": total_params,
            "calculation": {
                "gates": f"LSTM有{num_gates}个门：输入门、遗忘门、细胞门、输出门",
                "first_layer": f"第一层参数: {num_gates} × ({sequence_length} × {hidden_size} + {hidden_size}²) = {first_layer_params:,}",
                "other_layers": f"其他层参数: {num_gates} × ({hidden_size}² + {hidden_size}²) = {other_layer_params:,}",
                "total": f"总参数量: {total_params:,}",
            },
        }


def get_dynamic_example(example_type: str) -> Dict[str, Any]:
    """获取动态示例

    Args:
        example_type: 示例类型 ('cnn', 'vit', 'gnn', 'math', 'lstm')

    Returns:
        包含示例数据的字典
    """
    generator = ExampleGenerator()

    if example_type == "cnn":
        return generator.generate_cnn_example()
    elif example_type == "vit":
        return generator.generate_vit_example()
    elif example_type == "gnn":
        return generator.generate_gnn_example()
    elif example_type == "math":
        return generator.generate_math_example()
    elif example_type == "lstm":
        return generator.generate_lstm_example()
    else:
        raise ValueError(f"不支持的示例类型: {example_type}")
