"""
工具模块包

提供网络分析、可视化、缓存等核心功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

# 核心配置
from .config import CHINESE_SUPPORTED
from .i18n import get_text

# 核心计算模块
from .core import NetworkAnalyzer, ParameterCalculator, MemoryAnalyzer

# 可视化模块
from .visualization import ChartBuilder, PlotHelper

# 异常处理
from .exceptions import (
    NetworkAnalysisError,
    InvalidLayerConfigError,
    InsufficientMemoryError,
    ComputationError,
    VisualizationError,
    ModelLoadError,
    ConfigurationError,
    DataValidationError,
    CacheError,
    handle_exceptions,
)

# 缓存管理
from .cache import (
    CacheManager,
    get_cache_manager,
    cached,
    cached_method,
    network_analysis_key,
    param_calculation_key,
)

__all__ = [
    # 配置
    "CHINESE_SUPPORTED",
    "get_text",
    # 核心计算
    "NetworkAnalyzer",
    "ParameterCalculator",
    "MemoryAnalyzer",
    # 可视化
    "ChartBuilder",
    "PlotHelper",
    # 异常处理
    "NetworkAnalysisError",
    "InvalidLayerConfigError",
    "InsufficientMemoryError",
    "ComputationError",
    "VisualizationError",
    "ModelLoadError",
    "ConfigurationError",
    "DataValidationError",
    "CacheError",
    "handle_exceptions",
    # 缓存管理
    "CacheManager",
    "get_cache_manager",
    "cached",
    "cached_method",
    "network_analysis_key",
    "param_calculation_key",
]
