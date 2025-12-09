"""
异常处理系统单元测试
"""

import pytest
from utils.exceptions import (
    NetworkAnalysisError,
    InvalidLayerConfigError,
    ComputationError,
    DataValidationError,
    CacheError,
    exception_handler,
)


class TestCustomExceptions:
    """自定义异常测试类"""

    def test_network_analysis_error(self):
        """测试网络分析错误"""
        with pytest.raises(NetworkAnalysisError) as exc_info:
            raise NetworkAnalysisError("网络结构无效")

        assert str(exc_info.value) == "网络结构无效"
        assert exc_info.value.__class__.__name__ == "NetworkAnalysisError"

    def test_invalid_layer_config_error(self):
        """测试无效层配置错误"""
        with pytest.raises(InvalidLayerConfigError) as exc_info:
            raise InvalidLayerConfigError("Conv2d", "kernel_size", -1, "正整数")

        assert "Conv2d" in str(exc_info.value)
        assert "kernel_size" in str(exc_info.value)

    def test_computation_error(self):
        """测试计算错误"""
        with pytest.raises(ComputationError) as exc_info:
            raise ComputationError(
                operation="矩阵乘法", error_details="维度不匹配: (3,4) @ (5,6)"
            )

        assert "矩阵乘法" in str(exc_info.value)
        assert "维度不匹配" in str(exc_info.value)

    def test_data_validation_error(self):
        """测试数据验证错误"""
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError(
                data_name="input_shape", 
                actual_value="(H,W,C)", 
                expected_condition="(C,H,W)格式"
            )

        assert "input_shape" in str(exc_info.value)
        assert "(C,H,W)" in str(exc_info.value)
        assert "(H,W,C)" in str(exc_info.value)

    def test_cache_error(self):
        """测试缓存错误"""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("set", "test_key", "序列化失败")

        assert "缓存" in str(exc_info.value)
        assert "test_key" in str(exc_info.value)


class TestExceptionHandler:
    """异常处理器测试类"""

    def test_exception_decorator_success(self):
        """测试异常处理装饰器 - 成功情况"""

        @exception_handler
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_exception_decorator_catches_network_error(self):
        """测试异常处理装饰器 - 捕获网络错误"""

        @exception_handler()
        def failing_function():
            raise NetworkAnalysisError("测试错误")

        # 应该返回None（默认返回值）
        result = failing_function()
        assert result is None

    def test_exception_decorator_catches_generic_error(self):
        """测试异常处理装饰器 - 捕获通用错误"""

        @exception_handler()
        def generic_error_function():
            raise ValueError("通用错误")

        result = generic_error_function()
        assert result is None

    def test_exception_decorator_with_reraise(self):
        """测试异常处理装饰器 - 重新抛出异常"""

        @exception_handler(reraise=True)
        def reraise_function():
            raise NetworkAnalysisError("重新抛出错误")

        with pytest.raises(NetworkAnalysisError):
            reraise_function()

    def test_exception_decorator_with_logging(self, caplog):
        """测试异常处理装饰器 - 日志记录"""

        @exception_handler(log_errors=True)
        def logging_error_function():
            raise NetworkAnalysisError("日志测试错误")

        result = logging_error_function()

        # 验证返回None
        assert result is None

        # 验证日志记录
        assert "日志测试错误" in caplog.text

    def test_exception_preserves_traceback(self):
        """测试异常处理保留堆栈跟踪"""

        @exception_handler()
        def deep_function():
            return intermediate_function()

        def intermediate_function():
            raise NetworkAnalysisError("深层错误")

        result = deep_function()

        # 异常被捕获，返回默认值
        assert result is None
