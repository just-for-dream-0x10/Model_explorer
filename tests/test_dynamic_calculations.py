"""
动态计算测试脚本
Test Script for Dynamic Calculations

验证所有硬编码修复后的动态计算是否正确工作
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_performance_predictor():
    """测试性能预测器"""
    print("🔍 测试性能预测器...")

    try:
        from utils.performance_predictor import (
            PerformancePredictor,
            create_model_config,
            create_dataset_config,
            create_training_config,
        )

        predictor = PerformancePredictor()

        # 测试CNN模型
        model_config = create_model_config("CNN", 5e6, 10)
        dataset_config = create_dataset_config(50000, 10)
        training_config = create_training_config(0.001, 32, 100)

        curves = predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert len(curves["train_loss"]) == 100
        assert len(curves["val_acc"]) == 100
        assert curves["final_val_acc"] > 0
        print("✅ 性能预测器测试通过")

    except Exception as e:
        print(f"❌ 性能预测器测试失败: {e}")
        return False

    return True


def test_example_generator():
    """测试示例生成器"""
    print("🔍 测试示例生成器...")

    try:
        from utils.example_generator import get_dynamic_example

        # 测试CNN示例
        cnn_example = get_dynamic_example("cnn")
        assert "input_size" in cnn_example
        assert "kernel_size" in cnn_example
        assert "output_size" in cnn_example

        # 测试ViT示例
        vit_example = get_dynamic_example("vit")
        assert "img_size" in vit_example
        assert "num_patches" in vit_example
        assert "d_model" in vit_example

        # 测试GNN示例
        gnn_example = get_dynamic_example("gnn")
        assert "num_nodes" in gnn_example
        assert "feature_dim" in gnn_example

        print("✅ 示例生成器测试通过")

    except Exception as e:
        print(f"❌ 示例生成器测试失败: {e}")
        return False

    return True


def test_template_calculator():
    """测试模板计算器"""
    print("🔍 测试模板计算器...")

    try:
        from utils.template_calculator import TemplateCalculator

        # 测试flatten计算
        shape = (64, 32, 32)
        flattened = TemplateCalculator.calculate_flattened_size(shape)
        assert flattened == 64 * 32 * 32

        # 测试特征数建议
        fc_features = TemplateCalculator.suggest_fc_features(10000, 10)
        assert len(fc_features) >= 2
        assert fc_features[-1] == 10

        # 测试通道数建议
        conv_channels = TemplateCalculator.suggest_conv_channels((3, 32, 32))
        assert len(conv_channels) >= 3

        # 测试MNIST模板
        mnist_template = TemplateCalculator.create_mnist_template((1, 1, 28, 28))
        assert len(mnist_template) > 0

        # 测试CIFAR模板
        cifar_template = TemplateCalculator.create_cifar_template((1, 3, 32, 32))
        assert len(cifar_template) > 0

        # 测试MLP模板
        mlp_template = TemplateCalculator.create_mlp_template((1, 784))
        assert len(mlp_template) > 0

        print("✅ 模板计算器测试通过")

    except Exception as e:
        print(f"❌ 模板计算器测试失败: {e}")
        return False

    return True


def test_parameter_suggester():
    """测试参数建议器"""
    print("🔍 测试参数建议器...")

    try:
        from utils.parameter_suggester import get_suggested_params

        # 测试GNN参数建议
        gnn_params = get_suggested_params(
            "gnn", num_nodes=100, feature_dim=16, task_complexity="medium"
        )
        assert "num_layers" in gnn_params
        assert "hidden_dims" in gnn_params

        # 测试RNN参数建议
        rnn_params = get_suggested_params(
            "rnn", sequence_length=50, input_size=32, task_type="classification"
        )
        assert "hidden_size" in rnn_params
        assert "num_layers" in rnn_params

        # 测试ViT参数建议
        vit_params = get_suggested_params(
            "vit", img_size=224, num_classes=10, model_size="base"
        )
        assert "embed_dim" in vit_params
        assert "num_heads" in vit_params

        # 测试归一化参数建议
        norm_params = get_suggested_params(
            "normalization", input_shape=(1, 64, 32, 32), batch_size=32
        )
        assert "recommended_norm" in norm_params

        print("✅ 参数建议器测试通过")

    except Exception as e:
        print(f"❌ 参数建议器测试失败: {e}")
        return False

    return True


def test_dynamic_updates():
    """测试动态更新"""
    print("🔍 测试动态更新...")

    try:
        # 模拟用户参数变化
        test_cases = [
            {"input_size": 224, "kernel_size": 3, "expected_output": 112},
            {"input_size": 32, "kernel_size": 3, "expected_output": 30},
            {"input_size": 64, "kernel_size": 5, "expected_output": 12},
        ]

        for case in test_cases:
            input_size = case["input_size"]
            kernel_size = case["kernel_size"]
            expected = case["expected_output"]

            # 计算输出尺寸
            output_size = (input_size - kernel_size) // 1 + 1
            assert output_size == expected, f"期望 {expected}, 得到 {output_size}"

        print("✅ 动态更新测试通过")

    except Exception as e:
        print(f"❌ 动态更新测试失败: {e}")
        return False

    return True


def test_integration():
    """测试集成功能"""
    print("🔍 测试集成功能...")

    try:
        # 测试性能预测器和参数建议器的集成
        from utils.performance_predictor import PerformancePredictor
        from utils.parameter_suggester import get_suggested_params

        # 获取ViT参数建议
        vit_params = get_suggested_params(
            "vit", img_size=224, num_classes=10, model_size="base"
        )

        # 使用建议的参数进行性能预测
        predictor = PerformancePredictor()
        model_config = {
            "model_type": "Transformer",
            "num_params": vit_params["embed_dim"]
            * vit_params["embed_dim"]
            * 4,  # 粗略估算
            "model_depth": vit_params["num_layers"],
            "model_complexity": 0.8,
        }

        dataset_config = {
            "dataset_size": 50000,
            "num_classes": 10,
            "data_complexity": 0.5,
        }

        training_config = {"learning_rate": 0.001, "batch_size": 32, "num_epochs": 100}

        curves = predictor.predict_training_performance(
            model_config, dataset_config, training_config
        )

        assert curves["final_val_acc"] > 0.5  # ViT应该能达到50%以上精度

        print("✅ 集成功能测试通过")

    except Exception as e:
        print(f"❌ 集成功能测试失败: {e}")
        return False

    return True


def main():
    """主测试函数"""
    print("🚀 开始动态计算测试...")
    print("=" * 50)

    tests = [
        test_performance_predictor,
        test_example_generator,
        test_template_calculator,
        test_parameter_suggester,
        test_dynamic_updates,
        test_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！动态计算功能正常工作。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
