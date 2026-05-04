"""
模型对比工具测试
Tests for utils/model_comparison.py
"""

import pytest
import numpy as np
from utils.model_comparison import (
    get_model_info,
    generate_training_curves,
    compare_convergence_speed,
    get_data_efficiency_curve,
    get_comparison_recommendations,
)


class TestGetModelInfo:
    """模型信息获取测试"""

    def test_resnet18_info(self):
        """测试ResNet-18信息"""
        info = get_model_info("ResNet-18")

        assert info["type"] == "CNN"
        assert info["params"] == 11.7
        assert info["flops"] == 1.8
        assert info["depth"] == 18
        assert "architecture" in info

    def test_vit_base_info(self):
        """测试ViT-Base信息"""
        info = get_model_info("ViT-Base")

        assert info["type"] == "Transformer"
        assert info["params"] == 86.0
        assert info["depth"] == 12

    def test_unknown_model_raises(self):
        """测试未知模型抛出异常"""
        with pytest.raises(ValueError, match="未知模型"):
            get_model_info("NonExistentModel")

    def test_all_known_models(self):
        """测试所有已知模型都能正常获取"""
        known_models = [
            "ResNet-18", "ResNet-50", "MobileNet-V2",
            "ViT-Tiny", "ViT-Small", "ViT-Base",
        ]
        for model_name in known_models:
            info = get_model_info(model_name)
            assert "type" in info
            assert "params" in info
            assert "flops" in info


class TestGenerateTrainingCurves:
    """训练曲线生成测试"""

    def test_cnn_curves(self):
        """测试CNN训练曲线"""
        curves = generate_training_curves(
            model_type="CNN", dataset_size="medium", num_epochs=50
        )

        assert "train_loss" in curves
        assert "val_loss" in curves
        assert "train_acc" in curves
        assert "val_acc" in curves
        assert len(curves["train_loss"]) == 50

    def test_transformer_curves(self):
        """测试Transformer训练曲线"""
        curves = generate_training_curves(
            model_type="Transformer", dataset_size="large", num_epochs=100
        )

        assert len(curves["train_loss"]) == 100
        assert curves["final_val_acc"] > 0

    def test_rnn_curves(self):
        """测试RNN训练曲线"""
        curves = generate_training_curves(
            model_type="RNN", dataset_size="small", num_epochs=30
        )

        assert len(curves["train_loss"]) == 30

    def test_seed_reproducibility(self):
        """测试随机种子可复现性"""
        curves1 = generate_training_curves(
            model_type="CNN", dataset_size="medium", num_epochs=20, seed=42
        )
        curves2 = generate_training_curves(
            model_type="CNN", dataset_size="medium", num_epochs=20, seed=42
        )

        assert curves1["train_loss"] == curves2["train_loss"]

    def test_accuracy_in_valid_range(self):
        """测试精度在有效范围内"""
        curves = generate_training_curves(
            model_type="CNN", dataset_size="medium", num_epochs=50
        )

        assert all(0 <= acc <= 1 for acc in curves["train_acc"])
        assert all(0 <= acc <= 1 for acc in curves["val_acc"])

    def test_loss_decreases(self):
        """测试损失逐渐下降"""
        curves = generate_training_curves(
            model_type="CNN", dataset_size="medium", num_epochs=50
        )

        # 最终损失应小于初始损失
        assert curves["train_loss"][-1] < curves["train_loss"][0]


class TestCompareConvergenceSpeed:
    """收敛速度对比测试"""

    def test_basic_comparison(self):
        """测试基本收敛速度对比"""
        curves_dict = {
            "CNN": generate_training_curves("CNN", "medium", num_epochs=50),
            "Transformer": generate_training_curves("Transformer", "medium", num_epochs=50),
        }

        comparison = compare_convergence_speed(curves_dict)

        assert "CNN" in comparison
        assert "Transformer" in comparison
        assert "best_acc" in comparison["CNN"]
        assert "epoch_to_90" in comparison["CNN"]
        assert "epoch_to_95" in comparison["CNN"]

    def test_epoch_values_valid(self):
        """测试epoch值有效"""
        curves_dict = {
            "Model_A": generate_training_curves("CNN", "medium", num_epochs=50),
        }

        comparison = compare_convergence_speed(curves_dict)

        assert 0 <= comparison["Model_A"]["epoch_to_90"] <= 50
        assert 0 <= comparison["Model_A"]["epoch_to_95"] <= 50


class TestGetDataEfficiencyCurve:
    """数据效率曲线测试"""

    def test_cnn_efficiency(self):
        """测试CNN数据效率曲线"""
        curve = get_data_efficiency_curve("CNN")

        assert "data_ratios" in curve
        assert "accuracies" in curve
        assert len(curve["data_ratios"]) == 6
        assert len(curve["accuracies"]) == 6

    def test_transformer_efficiency(self):
        """测试Transformer数据效率曲线"""
        curve = get_data_efficiency_curve("Transformer")

        assert len(curve["accuracies"]) == 6

    def test_accuracy_monotonic_increase(self):
        """测试精度随数据量增加而提升"""
        curve = get_data_efficiency_curve("CNN")

        # 精度应该总体呈上升趋势
        assert curve["accuracies"][-1] > curve["accuracies"][0]

    def test_accuracy_in_valid_range(self):
        """测试精度在有效范围内"""
        for model_type in ["CNN", "Transformer"]:
            curve = get_data_efficiency_curve(model_type)
            for acc in curve["accuracies"]:
                assert 0 <= acc <= 1


class TestGetComparisonRecommendations:
    """模型推荐测试"""

    def test_small_data_low_budget(self):
        """测试小数据+低算力推荐"""
        rec = get_comparison_recommendations("small", "low", "classification")

        assert rec["primary"] == "MobileNet-V2"
        assert rec["alternative"] == "ResNet-18"

    def test_large_data_high_budget(self):
        """测试大数据+高算力推荐"""
        rec = get_comparison_recommendations("large", "high", "classification")

        assert rec["primary"] == "ViT-Base"

    def test_medium_data(self):
        """测试中等数据推荐"""
        rec = get_comparison_recommendations("medium", "medium", "classification")

        assert rec["primary"] in ["ResNet-50", "ViT-Small"]
        assert rec["reason"] != ""

    def test_all_combinations_have_results(self):
        """测试所有组合都有推荐结果"""
        for data_size in ["small", "medium", "large"]:
            for compute_budget in ["low", "medium", "high"]:
                rec = get_comparison_recommendations(
                    data_size, compute_budget, "classification"
                )
                assert rec["primary"] is not None
                assert rec["alternative"] is not None
