"""自定义异常类

定义项目中使用的所有自定义异常，提供更精确的错误处理。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Optional, Any


class NetworkAnalysisError(Exception):
    """网络分析相关异常基类

    所有网络分析相关的异常都应该继承自这个基类。
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """初始化异常

        Args:
            message: 错误信息
            details: 错误详细信息字典
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """返回格式化的错误信息"""
        if self.details:
            details_str = ", ".join(f"{k}: {v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class InvalidLayerConfigError(NetworkAnalysisError):
    """无效层配置异常

    当网络层的配置参数无效时抛出。
    """

    def __init__(
        self,
        layer_type: str,
        param_name: str,
        param_value: Any,
        expected_type: Optional[str] = None,
    ):
        """初始化异常

        Args:
            layer_type: 层类型
            param_name: 参数名
            param_value: 参数值
            expected_type: 期望的类型
        """
        message = f"{layer_type}层的{param_name}参数无效: {param_value}"
        details = {
            "layer_type": layer_type,
            "param_name": param_name,
            "param_value": str(param_value),
            "param_type": type(param_value).__name__,
        }

        if expected_type:
            details["expected_type"] = expected_type
            message += f" (期望类型: {expected_type})"

        super().__init__(message, details)


class InsufficientMemoryError(NetworkAnalysisError):
    """内存不足异常

    当计算所需内存超过可用内存时抛出。
    """

    def __init__(
        self, required_memory_mb: float, available_memory_mb: Optional[float] = None
    ):
        """初始化异常

        Args:
            required_memory_mb: 所需内存(MB)
            available_memory_mb: 可用内存(MB)
        """
        message = f"内存不足，需要{required_memory_mb:.1f}MB"
        details = {"required_memory_mb": required_memory_mb}

        if available_memory_mb:
            details["available_memory_mb"] = available_memory_mb
            message += f"，可用{available_memory_mb:.1f}MB"

        super().__init__(message, details)


class ComputationError(NetworkAnalysisError):
    """计算错误异常

    当数值计算过程中出现错误时抛出。
    """

    def __init__(self, operation: str, error_details: Optional[str] = None):
        """初始化异常

        Args:
            operation: 操作名称
            error_details: 错误详情
        """
        message = f"计算错误: {operation}"
        if error_details:
            message += f" - {error_details}"

        details = {"operation": operation}
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


class VisualizationError(NetworkAnalysisError):
    """可视化错误异常

    当图表创建或显示过程中出现错误时抛出。
    """

    def __init__(self, chart_type: str, error_details: Optional[str] = None):
        """初始化异常

        Args:
            chart_type: 图表类型
            error_details: 错误详情
        """
        message = f"可视化错误: {chart_type}"
        if error_details:
            message += f" - {error_details}"

        details = {"chart_type": chart_type}
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


class ModelLoadError(NetworkAnalysisError):
    """模型加载错误异常

    当加载预定义模型模板时出现错误时抛出。
    """

    def __init__(self, model_id: str, error_details: Optional[str] = None):
        """初始化异常

        Args:
            model_id: 模型ID
            error_details: 错误详情
        """
        message = f"模型加载失败: {model_id}"
        if error_details:
            message += f" - {error_details}"

        details = {"model_id": model_id}
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


class ConfigurationError(NetworkAnalysisError):
    """配置错误异常

    当系统配置或参数配置出现错误时抛出。
    """

    def __init__(
        self, config_key: str, config_value: Any, error_details: Optional[str] = None
    ):
        """初始化异常

        Args:
            config_key: 配置键
            config_value: 配置值
            error_details: 错误详情
        """
        message = f"配置错误: {config_key} = {config_value}"
        if error_details:
            message += f" - {error_details}"

        details = {
            "config_key": config_key,
            "config_value": str(config_value),
            "config_type": type(config_value).__name__,
        }
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


class DataValidationError(NetworkAnalysisError):
    """数据验证错误异常

    当输入数据不符合预期格式或范围时抛出。
    """

    def __init__(
        self,
        data_name: str,
        actual_value: Any,
        expected_condition: str,
        error_details: Optional[str] = None,
    ):
        """初始化异常

        Args:
            data_name: 数据名称
            actual_value: 实际值
            expected_condition: 期望条件
            error_details: 错误详情
        """
        message = (
            f"数据验证失败: {data_name} = {actual_value}，期望{expected_condition}"
        )
        if error_details:
            message += f" - {error_details}"

        details = {
            "data_name": data_name,
            "actual_value": str(actual_value),
            "actual_type": type(actual_value).__name__,
            "expected_condition": expected_condition,
        }
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


class CacheError(NetworkAnalysisError):
    """缓存错误异常

    当缓存操作出现错误时抛出。
    """

    def __init__(
        self, operation: str, cache_key: str, error_details: Optional[str] = None
    ):
        """初始化异常

        Args:
            operation: 操作类型 (get/set/delete)
            cache_key: 缓存键
            error_details: 错误详情
        """
        message = f"缓存{operation}失败: {cache_key}"
        if error_details:
            message += f" - {error_details}"

        details = {"operation": operation, "cache_key": cache_key}
        if error_details:
            details["error_details"] = error_details

        super().__init__(message, details)


# 异常处理装饰器
def handle_exceptions(
    default_return: Any = None, log_errors: bool = True, reraise: bool = False
):
    """异常处理装饰器

    Args:
        default_return: 异常时的默认返回值
        log_errors: 是否记录错误日志
        reraise: 是否重新抛出异常

    Returns:
        装饰器函数
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except NetworkAnalysisError as e:
                if log_errors:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(f"网络分析错误: {e}", extra=e.details)

                if reraise:
                    raise
                return default_return
            except Exception as e:
                if log_errors:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(f"未知错误: {e}")

                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator
