"""
动态示例生成器测试
Tests for utils/example_generator.py
"""

import pytest
import numpy as np
from utils.example_generator import ExampleGenerator, get_dynamic_example


class TestExampleGenerator:
    """动态示例生成器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.generator = ExampleGenerator()

    def test_cnn_example_structure(self):
        """测试CNN示例结构"""
        example = self.generator.generate_cnn_example()

        assert "input_size" in example
        assert "kernel_size" in example
        assert "stride" in example
        assert "padding" in example
        assert "output_size" in example
        assert "input_matrix" in example
        assert "kernel" in example
        assert "calculation_formula" in example

    def test_cnn_output_size_calculation(self):
        """测试CNN输出尺寸计算"""
        example = self.generator.generate_cnn_example()

        # 验证输出尺寸计算正确
        expected = (
            example["input_size"]
            + 2 * example["padding"]
            - example["kernel_size"]
        ) // example["stride"] + 1
        assert example["output_size"] == expected

    def test_cnn_matrix_shapes(self):
        """测试CNN矩阵形状"""
        example = self.generator.generate_cnn_example()

        input_matrix = np.array(example["input_matrix"])
        kernel = np.array(example["kernel"])

        assert input_matrix.shape == (example["input_size"], example["input_size"])
        assert kernel.shape == (example["kernel_size"], example["kernel_size"])

    def test_vit_example_structure(self):
        """测试ViT示例结构"""
        example = self.generator.generate_vit_example()

        assert "img_size" in example
        assert "patch_size" in example
        assert "num_patches" in example
        assert "seq_len" in example
        assert "d_model" in example
        assert "num_heads" in example
        assert "attn_matrix_size" in example
        assert "calculation" in example

    def test_vit_patch_calculation(self):
        """测试ViT patch数量计算"""
        example = self.generator.generate_vit_example()

        expected_patches = (example["img_size"] // example["patch_size"]) ** 2
        assert example["num_patches"] == expected_patches
        assert example["seq_len"] == expected_patches + 1  # +1 for CLS token

    def test_gnn_example_structure(self):
        """测试GNN示例结构"""
        example = self.generator.generate_gnn_example()

        assert "num_nodes" in example
        assert "feature_dim" in example
        assert "adj_matrix" in example
        assert "node_features" in example
        assert "weight_matrix" in example
        assert "calculation" in example

    def test_gnn_adjacency_matrix_symmetric(self):
        """测试GNN邻接矩阵对称性"""
        example = self.generator.generate_gnn_example()

        adj = np.array(example["adj_matrix"])
        assert np.array_equal(adj, adj.T), "邻接矩阵应该是对称的"

    def test_gnn_adjacency_self_loops(self):
        """测试GNN邻接矩阵有自环"""
        example = self.generator.generate_gnn_example()

        adj = np.array(example["adj_matrix"])
        for i in range(example["num_nodes"]):
            assert adj[i, i] == 1, "对角线应该为1（自环）"

    def test_gnn_feature_shape(self):
        """测试GNN特征矩阵形状"""
        example = self.generator.generate_gnn_example()

        features = np.array(example["node_features"])
        assert features.shape == (example["num_nodes"], example["feature_dim"])

    def test_math_example_structure(self):
        """测试数学推导示例结构"""
        example = self.generator.generate_math_example()

        assert "num_nodes" in example
        assert "adj_matrix" in example
        assert "degree_matrix" in example
        assert "laplacian" in example
        assert "formulas" in example

    def test_math_laplacian_calculation(self):
        """测试拉普拉斯矩阵计算"""
        example = self.generator.generate_math_example()

        adj = np.array(example["adj_matrix"])
        degree = np.array(example["degree_matrix"])
        laplacian = np.array(example["laplacian"])

        expected_laplacian = degree - adj
        np.testing.assert_array_almost_equal(laplacian, expected_laplacian)

    def test_lstm_example_structure(self):
        """测试LSTM示例结构"""
        example = self.generator.generate_lstm_example()

        assert "hidden_size" in example
        assert "num_layers" in example
        assert "sequence_length" in example
        assert "num_gates" in example
        assert "total_params" in example
        assert "calculation" in example

    def test_lstm_gate_count(self):
        """测试LSTM门数量"""
        example = self.generator.generate_lstm_example()

        assert example["num_gates"] == 4  # LSTM有4个门

    def test_lstm_total_params(self):
        """测试LSTM总参数量计算"""
        example = self.generator.generate_lstm_example()

        expected_total = (
            example["first_layer_params"]
            + example["other_layer_params"] * (example["num_layers"] - 1)
        )
        assert example["total_params"] == expected_total


class TestGetDynamicExample:
    """get_dynamic_example 函数测试"""

    def test_cnn_type(self):
        """测试CNN类型"""
        example = get_dynamic_example("cnn")
        assert "input_matrix" in example
        assert "kernel" in example

    def test_vit_type(self):
        """测试ViT类型"""
        example = get_dynamic_example("vit")
        assert "num_patches" in example
        assert "seq_len" in example

    def test_gnn_type(self):
        """测试GNN类型"""
        example = get_dynamic_example("gnn")
        assert "adj_matrix" in example
        assert "node_features" in example

    def test_math_type(self):
        """测试数学推导类型"""
        example = get_dynamic_example("math")
        assert "laplacian" in example

    def test_lstm_type(self):
        """测试LSTM类型"""
        example = get_dynamic_example("lstm")
        assert "total_params" in example

    def test_unknown_type_raises(self):
        """测试未知类型抛出异常"""
        with pytest.raises(ValueError, match="不支持的示例类型"):
            get_dynamic_example("unknown_type")

    def test_all_types_return_dict(self):
        """测试所有类型都返回字典"""
        for example_type in ["cnn", "vit", "gnn", "math", "lstm"]:
            example = get_dynamic_example(example_type)
            assert isinstance(example, dict)
