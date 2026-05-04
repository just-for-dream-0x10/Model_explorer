"""
性能预测器测试
Tests for utils/performance_predictor.py
"""

import pytest
import numpy as np
from utils.performance_predictor import (
    PerformancePredictor,
    create_model_config,
    create_dataset_config,
    create_training_config,
)


class TestPerformancePredictor:
    """性能预测器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.predictor = PerformancePredictor()

    def test_basic_prediction(self):
        """测试基本预测功能"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
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
        for epochs in [10, 50, 100, 200]:
            training_config = create_training_config(num_epochs=epochs)
            model_config = create_model_config("CNN", 5e6, 10)
            dataset_config = create_dataset_config(50000, 10)

            result = self.predictor.predict_training_performance(
                model_config, dataset_config, training_config
            )

            assert len(result["train_loss"]) == epochs
            assert len(result["val_loss"]) == epochs
            assert len(result["train_acc"]) == epochs
            assert len(result["val_acc"]) == epochs

    def test_loss_positive(self):
        """测试损失值始终为正"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert all(loss > 0 for loss in result["train_loss"])
        assert all(loss > 0 for loss in result["val_loss"])

    def test_accuracy_in_range(self):
        """测试精度在 [0, 1] 范围内"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert all(0 <= acc <= 1 for acc in result["train_acc"])
        assert all(0 <= acc <= 1 for acc in result["val_acc"])

    def test_loss_decreases_over_time(self):
        """测试损失随时间下降"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        # 最终损失应小于初始损失
        assert result["train_loss"][-1] < result["train_loss"][0]

    def test_best_acc_gte_final_acc(self):
        """测试最佳精度 >= 最终精度"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert result["best_val_acc"] >= result["final_val_acc"]

    def test_convergence_epoch_valid(self):
        """测试收敛epoch在有效范围内"""
        epochs = 100
        training_config = create_training_config(num_epochs=epochs)
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert 0 <= result["convergence_epoch"] < epochs

    def test_transformer_small_data_penalty(self):
        """测试Transformer在小数据集上的惩罚"""
        model_config = create_model_config("Transformer", 20e6, 12)

        # 小数据集
        small_dataset = create_dataset_config(5000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result_small = self.predictor.predict_training_performance(
            model_config, small_dataset, training_config
        )

        # 大数据集
        large_dataset = create_dataset_config(500000, 10)
        result_large = self.predictor.predict_training_performance(
            model_config, large_dataset, training_config
        )

        # 大数据集上最终损失应该更低（训练更充分）
        assert result_large["val_loss"][-1] < result_small["val_loss"][-1]

    def test_cnn_stable_across_dataset_sizes(self):
        """测试CNN在不同数据集大小上的稳定性"""
        model_config = create_model_config("CNN", 5e6, 10)
        training_config = create_training_config(0.001, 32, 100)

        results = []
        for size in [10000, 50000, 100000]:
            dataset_config = create_dataset_config(size, 10)
            result = self.predictor.predict_training_performance(
                model_config, dataset_config, training_config
            )
            results.append(result["best_val_acc"])

        # CNN应该在不同数据集大小上都表现合理
        for acc in results:
            assert 0.1 < acc < 1.0

    def test_complexity_factor_calculation(self):
        """测试复杂度因子计算"""
        # 大模型
        large_model = create_model_config("CNN", 1e9, 50)
        small_model = create_model_config("CNN", 1e6, 5)

        dataset_config = create_dataset_config(50000, 10)

        complexity_large = self.predictor._calculate_complexity_factor(
            large_model, dataset_config
        )
        complexity_small = self.predictor._calculate_complexity_factor(
            small_model, dataset_config
        )

        # 大模型复杂度应该更高
        assert complexity_large > complexity_small

    def test_val_loss_higher_than_train(self):
        """测试验证损失通常高于训练损失"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        result = self.predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        # 验证损失应该总体上高于训练损失
        avg_train_loss = np.mean(result["train_loss"])
        avg_val_loss = np.mean(result["val_loss"])
        assert avg_val_loss > avg_train_loss


class TestConfigCreators:
    """配置创建器测试"""

    def test_create_model_config(self):
        """测试模型配置创建"""
        config = create_model_config("CNN", 5e6, 10)

        assert config["model_type"] == "CNN"
        assert config["num_params"] == 5e6
        assert config["model_depth"] == 10
        assert "model_complexity" in config

    def test_create_dataset_config(self):
        """测试数据集配置创建"""
        config = create_dataset_config(50000, 10, 0.5)

        assert config["dataset_size"] == 50000
        assert config["num_classes"] == 10
        assert config["data_complexity"] == 0.5

    def test_create_training_config(self):
        """测试训练配置创建"""
        config = create_training_config(0.001, 32, 100)

        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["num_epochs"] == 100

    def test_default_configs(self):
        """测试默认配置"""
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config()

        assert training_config["learning_rate"] == 0.001
        assert training_config["batch_size"] == 32
        assert training_config["num_epochs"] == 100
