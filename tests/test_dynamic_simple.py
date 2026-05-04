"""
简化的动态计算测试脚本
Simplified Test Script for Dynamic Calculations

验证核心动态计算功能
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


def test_template_calculator():
    """测试模板计算器"""
    print("🔍 测试模板计算器...")

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

    print("✅ 模板计算器测试通过")


def test_parameter_suggester():
    """测试参数建议器"""
    print("🔍 测试参数建议器...")

    from utils.parameter_suggester import get_suggested_params

    # 测试GNN参数建议
    gnn_params = get_suggested_params(
        "gnn", num_nodes=100, feature_dim=16, task_complexity="medium"
    )
    assert "num_layers" in gnn_params
    assert "hidden_dims" in gnn_params

    # 测试ViT参数建议
    vit_params = get_suggested_params(
        "vit", img_size=224, num_classes=10, model_size="base"
    )
    assert "embed_dim" in vit_params
    assert "num_heads" in vit_params

    print("✅ 参数建议器测试通过")


def test_dynamic_calculations():
    """测试动态计算"""
    print("🔍 测试动态计算...")

    # 测试不同输入尺寸的输出计算
    test_cases = [
        {
            "input_size": 224,
            "kernel_size": 7,
            "stride": 2,
            "padding": 3,
            "expected": 112,
        },
        {
            "input_size": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "expected": 32,
        },
        {
            "input_size": 64,
            "kernel_size": 5,
            "stride": 2,
            "padding": 2,
            "expected": 32,
        },
    ]

    for case in test_cases:
        input_size = case["input_size"]
        kernel_size = case["kernel_size"]
        stride = case["stride"]
        padding = case["padding"]
        expected = case["expected"]

        # 计算输出尺寸
        output_size = (input_size + 2 * padding - kernel_size) // stride + 1
        assert output_size == expected, f"期望 {expected}, 得到 {output_size}"

    print("✅ 动态计算测试通过")


def main():
    """主测试函数"""
    print("🚀 开始动态计算测试...")
    print("=" * 50)

    tests = [
        test_performance_predictor,
        test_template_calculator,
        test_parameter_suggester,
        test_dynamic_calculations,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
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
