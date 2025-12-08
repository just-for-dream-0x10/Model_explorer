"""
动态计算验证测试用例
Dynamic Calculation Verification Test Cases

验证各个模块的动态计算是否正确
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DynamicCalculationTester:
    """动态计算测试器"""

    def __init__(self):
        self.test_results = []

    def test_conv2d_calculation(self):
        """测试Conv2d动态计算 - 对应参数计算器模块"""
        print("🔍 测试Conv2d动态计算...")

        from tabs.params_calculator import LayerAnalyzer

        # 测试用例：不同参数组合
        test_cases = [
            {
                "name": "标准卷积",
                "params": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 7,
                    "stride": 2,
                    "padding": 3,
                    "input_shape": (3, 224, 224),
                },
                "expected": {
                    "output_shape": (64, 112, 112),
                    "params_formula": "64 * 3 * 7 * 7 + 64 = 9472",
                },
            },
            {
                "name": "小卷积核",
                "params": {
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "input_shape": (64, 56, 56),
                },
                "expected": {
                    "output_shape": (128, 56, 56),
                    "params_formula": "128 * 64 * 3 * 3 + 128 = 73856",
                },
            },
            {
                "name": "大步长",
                "params": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                    "input_shape": (32, 128, 128),
                },
                "expected": {
                    "output_shape": (64, 64, 64),
                    "params_formula": "64 * 32 * 3 * 3 + 64 = 18496",
                },
            },
        ]

        analyzer = LayerAnalyzer()
        all_passed = True

        for case in test_cases:
            try:
                result = analyzer.conv2d_analysis(**case["params"])

                # 验证输出形状
                if result["output_shape"] != case["expected"]["output_shape"]:
                    print(f"❌ {case['name']}: 输出形状错误")
                    print(f"   期望: {case['expected']['output_shape']}")
                    print(f"   实际: {result['output_shape']}")
                    all_passed = False
                else:
                    print(f"✅ {case['name']}: 输出形状正确")

                # 验证参数量计算
                expected_params = eval(case["expected"]["params_formula"])
                if result["parameters"]["total"] != expected_params:
                    print(f"❌ {case['name']}: 参数量计算错误")
                    print(f"   期望: {expected_params}")
                    print(f"   实际: {result['parameters']['total']}")
                    all_passed = False
                else:
                    print(f"✅ {case['name']}: 参数量计算正确")

            except Exception as e:
                print(f"❌ {case['name']}: 测试失败 - {e}")
                all_passed = False

        self.test_results.append(
            {
                "module": "Conv2d计算 (参数计算器)",
                "status": "通过" if all_passed else "失败",
            }
        )
        return all_passed

    def test_vit_patch_calculation(self):
        """测试ViT patch计算 - 对应ViT分析模块"""
        print("\n🔍 测试ViT patch计算...")

        from utils.example_generator import get_dynamic_example

        # 测试不同图像尺寸和patch大小
        test_cases = [
            {
                "img_size": 224,
                "patch_size": 16,
                "expected_patches": 196,
                "expected_seq_len": 197,
            },
            {
                "img_size": 384,
                "patch_size": 16,
                "expected_patches": 576,
                "expected_seq_len": 577,
            },
            {
                "img_size": 224,
                "patch_size": 32,
                "expected_patches": 49,
                "expected_seq_len": 50,
            },
        ]

        all_passed = True

        for case in test_cases:
            try:
                # 模拟用户选择参数
                import streamlit as st

                if not hasattr(st, "session_state"):
                    st.session_state = {}
                st.session_state.vit_img_size = case["img_size"]
                st.session_state.vit_patch_size = case["patch_size"]

                example = get_dynamic_example("vit")

                if example["num_patches"] != case["expected_patches"]:
                    print(f"❌ ViT {case['img_size']}x{case['img_size']} patch数错误")
                    print(f"   期望: {case['expected_patches']}")
                    print(f"   实际: {example['num_patches']}")
                    all_passed = False
                else:
                    print(f"✅ ViT {case['img_size']}x{case['img_size']} patch数正确")

                if example["seq_len"] != case["expected_seq_len"]:
                    print(f"❌ ViT {case['img_size']}x{case['img_size']} 序列长度错误")
                    print(f"   期望: {case['expected_seq_len']}")
                    print(f"   实际: {example['seq_len']}")
                    all_passed = False
                else:
                    print(f"✅ ViT {case['img_size']}x{case['img_size']} 序列长度正确")

            except Exception as e:
                print(f"❌ ViT patch计算失败 - {e}")
                all_passed = False

        self.test_results.append(
            {
                "module": "ViT patch计算 (ViT分析)",
                "status": "通过" if all_passed else "失败",
            }
        )
        return all_passed

    def test_memory_calculation(self):
        """测试内存计算 - 对应内存分析模块"""
        print("\n🔍 测试内存计算...")

        from utils.memory_analyzer import analyze_conv2d_memory

        # 测试不同批次大小的内存占用
        test_cases = [
            {
                "name": "单批次",
                "params": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": (7, 7),
                    "input_shape": (1, 3, 224, 224),
                },
                "expected_ratio": 1.0,  # 基准
            },
            {
                "name": "大批次",
                "params": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": (7, 7),
                    "input_shape": (32, 3, 224, 224),
                },
                "expected_ratio": 32.0,  # 应该是32倍
            },
        ]

        all_passed = True

        for case in test_cases:
            try:
                info = analyze_conv2d_memory(**case["params"])

                # 验证内存与批次大小的关系
                if case["name"] == "大批次":
                    single_batch_info = analyze_conv2d_memory(
                        in_channels=3,
                        out_channels=64,
                        kernel_size=(7, 7),
                        input_shape=(1, 3, 224, 224),
                    )

                    ratio = info.backward_peak / single_batch_info.backward_peak

                    if abs(ratio - case["expected_ratio"]) > 0.1:
                        print(f"❌ {case['name']}: 内存比例错误")
                        print(f"   期望比例: {case['expected_ratio']}")
                        print(f"   实际比例: {ratio:.2f}")
                        all_passed = False
                    else:
                        print(f"✅ {case['name']}: 内存比例正确")
                else:
                    print(f"✅ {case['name']}: 基准测试通过")

            except Exception as e:
                print(f"❌ {case['name']}: 内存计算失败 - {e}")
                all_passed = False

        self.test_results.append(
            {
                "module": "内存计算 (内存分析)",
                "status": "通过" if all_passed else "失败",
            }
        )
        return all_passed

    def test_performance_prediction(self):
        """测试性能预测 - 对应模型对比模块"""
        print("\n🔍 测试性能预测...")

        from utils.performance_predictor import (
            PerformancePredictor,
            create_model_config,
            create_dataset_config,
            create_training_config,
        )

        # 测试不同模型配置的性能差异
        test_cases = [
            {
                "name": "小模型",
                "model_config": create_model_config("CNN", 1e6, 5),
                "expected_min_acc": 0.5,
                "expected_max_acc": 0.9,
            },
            {
                "name": "大模型",
                "model_config": create_model_config("CNN", 50e6, 20),
                "expected_min_acc": 0.7,
                "expected_max_acc": 0.95,
            },
            {
                "name": "Transformer小数据集",
                "model_config": create_model_config("Transformer", 20e6, 12),
                "dataset_config": create_dataset_config(10000, 10),  # 小数据集
                "expected_min_acc": 0.3,
                "expected_max_acc": 0.7,
            },
        ]

        predictor = PerformancePredictor()
        all_passed = True

        for case in test_cases:
            try:
                dataset_config = case.get(
                    "dataset_config", create_dataset_config(50000, 10)
                )
                training_config = create_training_config(0.001, 32, 50)

                curves = predictor.predict_training_performance(
                    case["model_config"], dataset_config, training_config
                )

                final_acc = curves["final_val_acc"]
                expected_min = case["expected_min_acc"]
                expected_max = case["expected_max_acc"]

                if not (expected_min <= final_acc <= expected_max):
                    print(f"❌ {case['name']}: 最终精度不在预期范围")
                    print(f"   预期范围: [{expected_min}, {expected_max}]")
                    print(f"   实际精度: {final_acc:.3f}")
                    all_passed = False
                else:
                    print(f"✅ {case['name']}: 最终精度在预期范围内")

            except Exception as e:
                print(f"❌ {case['name']}: 性能预测失败 - {e}")
                all_passed = False

        self.test_results.append(
            {
                "module": "性能预测 (模型对比)",
                "status": "通过" if all_passed else "失败",
            }
        )
        return all_passed

    def test_architecture_adaptation(self):
        """测试架构自适应 - 对应架构设计器模块"""
        print("\n🔍 测试架构自适应...")

        from utils.template_calculator import TemplateCalculator

        # 测试不同输入尺寸的模板适配
        test_cases = [
            {
                "name": "MNIST尺寸",
                "input_shape": (1, 1, 28, 28),
                "expected_flattened": 784,
            },
            {
                "name": "CIFAR尺寸",
                "input_shape": (1, 3, 32, 32),
                "expected_flattened": 32768,
            },
            {
                "name": "大图像",
                "input_shape": (1, 3, 64, 64),
                "expected_flattened": 131072,
            },
        ]

        calculator = TemplateCalculator()
        all_passed = True

        for case in test_cases:
            try:
                # 测试flatten计算
                after_conv_shape = (
                    64,
                    case["input_shape"][2] // 4,
                    case["input_shape"][3] // 4,
                )
                flattened = calculator.calculate_flattened_size(after_conv_shape)

                if flattened != case["expected_flattened"]:
                    print(f"❌ {case['name']}: flatten计算错误")
                    print(f"   期望: {case['expected_flattened']}")
                    print(f"   实际: {flattened}")
                    all_passed = False
                else:
                    print(f"✅ {case['name']}: flatten计算正确")

                # 测试FC特征数建议
                fc_features = calculator.suggest_fc_features(flattened, 10)
                if fc_features[-1] != 10:
                    print(f"❌ {case['name']}: FC特征数建议错误")
                    all_passed = False
                else:
                    print(f"✅ {case['name']}: FC特征数建议正确")

            except Exception as e:
                print(f"❌ {case['name']}: 架构自适应失败 - {e}")
                all_passed = False

        self.test_results.append(
            {
                "module": "架构自适应 (架构设计器)",
                "status": "通过" if all_passed else "失败",
            }
        )
        return all_passed

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始动态计算验证测试...")
        print("=" * 60)

        tests = [
            self.test_conv2d_calculation,
            self.test_vit_patch_calculation,
            self.test_memory_calculation,
            self.test_performance_prediction,
            self.test_architecture_adaptation,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1
            print()

        print("=" * 60)
        print("📊 测试结果汇总:")
        print()

        for result in self.test_results:
            status_icon = "✅" if result["status"] == "通过" else "❌"
            print(f"{status_icon} {result['module']}: {result['status']}")

        print()
        print(f"总计: {passed}/{total} 个模块测试通过")

        if passed == total:
            print("🎉 所有动态计算验证通过！")
            return True
        else:
            print("⚠️ 部分模块测试失败，需要修复。")
            return False


if __name__ == "__main__":
    tester = DynamicCalculationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
