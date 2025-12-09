"""
动态性能预测模块
Dynamic Performance Prediction Module

基于模型配置和数据集特征动态预测训练性能
替代硬编码的训练曲线数据
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class PerformancePredictor:
    """性能预测器

    基于模型配置、数据集大小等参数动态预测训练性能
    """

    def __init__(self):
        # 基准参数（基于实际训练经验）
        self.baseline_params = {
            "initial_loss_range": (2.0, 2.5),  # 初始损失范围
            "min_final_loss": 0.01,  # 最小最终损失
            "max_final_loss": 1.0,  # 最大最终损失
            "base_convergence_rate": 0.1,  # 基础收敛速度
            "noise_level": 0.05,  # 噪声水平
            "acc_improvement_factor": 0.8,  # 精度提升因子
        }

    def predict_training_performance(
        self, model_config: Dict, dataset_config: Dict, training_config: Dict
    ) -> Dict:
        """
        预测训练性能

        Args:
            model_config: 模型配置 {
                'model_type': 'CNN' | 'Transformer' | 'RNN',
                'num_params': 参数数量,
                'model_depth': 模型深度,
                'model_complexity': 复杂度评分
            }
            dataset_config: 数据集配置 {
                'dataset_size': 数据集大小,
                'num_classes': 类别数,
                'data_complexity': 数据复杂度
            }
            training_config: 训练配置 {
                'learning_rate': 学习率,
                'batch_size': 批次大小,
                'num_epochs': 训练轮数
            }

        Returns:
            包含预测性能的字典
        """
        # 计算模型复杂度因子
        complexity_factor = self._calculate_complexity_factor(
            model_config, dataset_config
        )

        # 预测损失曲线
        loss_curve = self._predict_loss_curve(
            model_config, dataset_config, training_config, complexity_factor
        )

        # 预测精度曲线
        acc_curve = self._predict_accuracy_curve(
            model_config, dataset_config, training_config, complexity_factor
        )

        # 计算收敛点
        convergence_epoch = self._calculate_convergence_epoch(loss_curve, acc_curve)

        return {
            "epochs": list(range(training_config["num_epochs"])),
            "train_loss": loss_curve["train"],
            "val_loss": loss_curve["val"],
            "train_acc": acc_curve["train"],
            "val_acc": acc_curve["val"],
            "final_val_acc": float(acc_curve["val"][-1]),
            "best_val_acc": float(np.max(acc_curve["val"])),
            "convergence_epoch": convergence_epoch,
            "complexity_factor": complexity_factor,
        }

    def _calculate_complexity_factor(
        self, model_config: Dict, dataset_config: Dict
    ) -> float:
        """计算复杂度因子"""
        # 模型复杂度（归一化到0-1）
        model_complexity = min(1.0, model_config.get("num_params", 1e6) / 1e9)

        # 数据复杂度（基于数据集大小和类别数）
        data_complexity = min(1.0, dataset_config.get("num_classes", 10) / 1000)
        data_size_factor = min(
            1.0, np.log10(dataset_config.get("dataset_size", 1000)) / 6
        )

        # 综合复杂度
        total_complexity = (
            model_complexity * 0.4 + data_complexity * 0.3 + data_size_factor * 0.3
        )

        return total_complexity

    def _predict_loss_curve(
        self,
        model_config: Dict,
        dataset_config: Dict,
        training_config: Dict,
        complexity_factor: float,
    ) -> Dict[str, List[float]]:
        """预测损失曲线"""
        epochs = training_config["num_epochs"]
        learning_rate = training_config["learning_rate"]

        # 基于复杂度计算初始损失
        base_initial = self.baseline_params["initial_loss_range"][0]
        initial_loss = base_initial + complexity_factor * 0.5

        # 基于模型类型和数据集计算最终损失
        model_type = model_config.get("model_type", "CNN")
        dataset_size = dataset_config.get("dataset_size", 10000)

        if model_type == "Transformer":
            # Transformer在小数据集上表现较差
            size_penalty = max(0, 1.0 - np.log10(dataset_size) / 6)
            final_loss = self.baseline_params["min_final_loss"] + size_penalty * 0.3
        else:
            # CNN/RNN相对稳定
            final_loss = (
                self.baseline_params["min_final_loss"] + (1 - complexity_factor) * 0.1
            )

        # 基于学习率调整收敛速度
        lr_factor = min(2.0, learning_rate / 0.001)
        convergence_speed = self.baseline_params["base_convergence_rate"] * lr_factor

        # 生成训练损失曲线（指数衰减 + 噪声）
        epoch_array = np.arange(epochs)
        train_loss = final_loss + (initial_loss - final_loss) * np.exp(
            -convergence_speed * epoch_array
        )

        # 添加噪声
        noise_level = self.baseline_params["noise_level"]
        train_loss += np.random.normal(0, noise_level, epochs)
        train_loss = np.clip(train_loss, final_loss * 0.5, initial_loss * 1.5)

        # 验证损失（略高于训练损失）
        val_loss = train_loss * (1.1 + complexity_factor * 0.1)
        val_loss += np.random.normal(0, noise_level * 0.5, epochs)
        val_loss = np.clip(val_loss, final_loss * 0.7, initial_loss * 1.8)

        return {"train": train_loss.tolist(), "val": val_loss.tolist()}

    def _predict_accuracy_curve(
        self,
        model_config: Dict,
        dataset_config: Dict,
        training_config: Dict,
        complexity_factor: float,
    ) -> Dict[str, List[float]]:
        """预测精度曲线"""
        epochs = training_config["num_epochs"]
        num_classes = dataset_config.get("num_classes", 10)
        dataset_size = dataset_config.get("dataset_size", 10000)
        model_type = model_config.get("model_type", "CNN")

        # 基于复杂度和数据集计算最终精度
        base_final_acc = 0.9  # 基础最终精度

        # 数据集大小影响
        size_factor = min(1.0, np.log10(dataset_size) / 5)

        # 模型类型影响
        type_penalty = 0.0
        type_bonus = 0.0

        if model_type == "Transformer":
            if dataset_size < 50000:
                # Transformer在小数据集上表现较差
                type_penalty = 0.2
            else:
                # Transformer在大数据集上表现优秀
                type_bonus = 0.05
        elif model_type == "CNN":
            # CNN表现稳定
            type_bonus = 0.0
        else:  # RNN
            # RNN通常精度稍低
            type_penalty = 0.1

        # 计算最终精度
        final_acc = (
            base_final_acc + size_factor * 0.08 - type_penalty * 0.1 + type_bonus * 0.05
        )
        final_acc = min(0.99, max(0.1, final_acc))

        # 初始精度（随机猜测）
        initial_acc = 1.0 / num_classes

        # 收敛速度（基于复杂度）
        convergence_speed = 0.1 + complexity_factor * 0.1

        # 生成训练精度曲线
        epoch_array = np.arange(epochs)
        train_acc = final_acc - (final_acc - initial_acc) * np.exp(
            -convergence_speed * epoch_array
        )

        # 添加噪声
        train_acc += np.random.normal(0, 0.02, epochs)
        train_acc = np.clip(train_acc, 0, 1)

        # 验证精度（略低于训练精度）
        val_acc = train_acc * (0.95 - complexity_factor * 0.05)
        val_acc += np.random.normal(0, 0.015, epochs)
        val_acc = np.clip(val_acc, 0, 1)

        return {"train": train_acc.tolist(), "val": val_acc.tolist()}

    def _calculate_convergence_epoch(self, loss_curve: Dict, acc_curve: Dict) -> int:
        """计算收敛点"""
        val_acc = np.array(acc_curve["val"])
        best_acc = np.max(val_acc)

        # 找到达到最佳精度95%的epoch
        target_acc = 0.95 * best_acc
        convergence_epochs = np.where(val_acc >= target_acc)[0]

        if len(convergence_epochs) > 0:
            return int(convergence_epochs[0])
        else:
            return len(val_acc) - 1  # 如果没收敛，返回最后一个epoch


def create_model_config(model_type: str, num_params: int, model_depth: int) -> Dict:
    """创建模型配置"""
    return {
        "model_type": model_type,
        "num_params": num_params,
        "model_depth": model_depth,
        "model_complexity": min(1.0, num_params / 1e9),
    }


def create_dataset_config(
    dataset_size: int, num_classes: int, data_complexity: float = 0.5
) -> Dict:
    """创建数据集配置"""
    return {
        "dataset_size": dataset_size,
        "num_classes": num_classes,
        "data_complexity": data_complexity,
    }


def create_training_config(
    learning_rate: float = 0.001, batch_size: int = 32, num_epochs: int = 100
) -> Dict:
    """创建训练配置"""
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }
