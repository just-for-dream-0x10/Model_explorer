#!/usr/bin/env python3
"""
简单测试运行器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cache_tests():
    """运行缓存测试"""
    try:
        from tests.test_cache import TestCacheManager
        import pytest

        # 创建测试实例
        test_instance = TestCacheManager()
        test_instance.setup_method()

        # 运行基本测试
        test_instance.test_basic_set_get()
        print("✓ 缓存基本功能测试通过")

        test_instance.test_cache_expiration()
        print("✓ 缓存过期测试通过")

        test_instance.test_lru_eviction()
        print("✓ LRU淘汰测试通过")

        test_instance.test_cached_decorator()
        print("✓ 缓存装饰器测试通过")

        return True

    except Exception as e:
        print(f"✗ 缓存测试失败: {e}")
        return False


def run_exception_tests():
    """运行异常处理测试"""
    try:
        from tests.test_exceptions import TestCustomExceptions, TestExceptionHandler

        # 创建测试实例
        exception_test = TestCustomExceptions()
        handler_test = TestExceptionHandler()

        # 运行异常测试
        exception_test.test_network_analysis_error()
        print("✓ 网络分析错误测试通过")

        exception_test.test_computation_error()
        print("✓ 计算错误测试通过")

        handler_test.test_exception_decorator_success()
        print("✓ 异常装饰器测试通过")

        return True

    except Exception as e:
        print(f"✗ 异常处理测试失败: {e}")
        return False


def run_layer_analyzer_tests():
    """运行层分析器测试"""
    try:
        from tests.test_layer_analyzer import TestLayerAnalyzer

        # 创建测试实例
        test_instance = TestLayerAnalyzer()
        test_instance.setup_method()

        # 运行基本测试
        test_instance.test_conv2d_analysis_basic()
        print("✓ Conv2d分析测试通过")

        test_instance.test_linear_analysis_basic()
        print("✓ Linear分析测试通过")

        test_instance.test_attention_analysis_basic()
        print("✓ 注意力分析测试通过")

        return True

    except Exception as e:
        print(f"✗ 层分析器测试失败: {e}")
        return False


def main():
    """主函数"""
    print("开始运行单元测试...")
    print("=" * 50)

    success_count = 0
    total_tests = 3

    if run_cache_tests():
        success_count += 1

    if run_exception_tests():
        success_count += 1

    if run_layer_analyzer_tests():
        success_count += 1

    print("=" * 50)
    print(f"测试结果: {success_count}/{total_tests} 通过")

    if success_count == total_tests:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
