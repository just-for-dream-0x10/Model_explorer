"""
训练模拟工具测试
Tests for utils/training.py
"""

import pytest
import numpy as np
from utils.training import simulate_training


class TestSimulateTraining:
    """训练模拟测试类"""

    def test_basic_simulation(self):
        """测试基本训练模拟"""
        result = simulate_training(
            epochs=50,
            model_type="CNN",
            num_params=5e6,
            num_classes=10,
            dataset_size=50000,
            learning_rate=0.001,
        )

        # 验证返回结构
        assert "train_loss" in result
        assert "val_loss" in result
        assert "train_acc" in result
        assert "val_acc" in result
        assert "epochs" in result
        assert "final_val_acc" in result
        assert "best_val_acc" in result
        assert "convergence_epoch" in result

    def test_curve_lengths(self):
        """测试曲线长度与epochs一致"""
        epochs = 30
        result = simulate_training(epochs=epochs, model_type="CNN")

        assert len(result["train_loss"]) == epochs
        assert len(result["val_loss"]) == epochs
        assert len(result["train_acc"]) == epochs
        assert len(result["val_acc"]) == epochs
        assert len(result["epochs"]) == epochs

    def test_loss_range(self):
        """测试损失值在合理范围内"""
        result = simulate_training(epochs=100, model_type="CNN")

        # 训练损失应该逐渐下降
        assert all(loss > 0 for loss in result["train_loss"])
        assert all(loss > 0 for loss in result["val_loss"])

        # 最终损失应该小于初始损失
        assert result["train_loss"][-1] < result["train_loss"][0]

    def test_accuracy_range(self):
        """测试精度值在合理范围内"""
        result = simulate_training(epochs=100, model_type="CNN")

        # 精度应该在 [0, 1] 范围内
        assert all(0 <= acc <= 1 for acc in result["train_acc"])
        assert all(0 <= acc <= 1 for acc in result["val_acc"])

    def test_best_acc_gte_final_acc(self):
        """测试最佳精度 >= 最终精度"""
        result = simulate_training(epochs=100, model_type="CNN")

        assert result["best_val_acc"] >= result["final_val_acc"]

    def test_convergence_epoch_in_range(self):
        """测试收敛epoch在合理范围内"""
        epochs = 100
        result = simulate_training(epochs=epochs, model_type="CNN")

        assert 0 <= result["convergence_epoch"] < epochs

    def test_different_model_types(self):
        """测试不同模型类型"""
        for model_type in ["CNN", "Transformer", "RNN"]:
            result = simulate_training(
                epochs=50, model_type=model_type, num_params=5e6
            )
            assert len(result["train_loss"]) == 50
            assert result["final_val_acc"] > 0

    def test_learning_rate_effect(self):
        """测试学习率对训练的影响"""
        result_low_lr = simulate_training(
            epochs=50, learning_rate=0.0001, model_type="CNN"
        )
        result_high_lr = simulate_training(
            epochs=50, learning_rate=0.01, model_type="CNN"
        )

        # 高学习率通常收敛更快，允许少量随机波动
        assert result_high_lr["convergence_epoch"] <= result_low_lr["convergence_epoch"] + 5

    def test_default_parameters(self):
        """测试默认参数"""
        result = simulate_training()

        assert "train_loss" in result
        assert len(result["train_loss"]) == 100  # 默认epochs=100

    def test_transformer_on_small_data(self):
        """测试Transformer在小数据集上的表现"""
        result = simulate_training(
            epochs=50,
            model_type="Transformer",
            dataset_size=5000,  # 小数据集
        )

        # Transformer在小数据集上精度应该较低
        assert result["final_val_acc"] < 0.95

    def test_cnn_on_large_data(self):
        """测试CNN在大数据集上的表现"""
        result = simulate_training(
            epochs=100,
            model_type="CNN",
            dataset_size=500000,
        )

        # CNN在大数据集上应该表现良好
        assert result["best_val_acc"] > 0.5
