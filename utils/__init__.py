"""
工具函数模块
"""

from .config import detect_chinese_support, configure_matplotlib_font, CHINESE_SUPPORTED
from .i18n import get_text
from .training import simulate_training

__all__ = [
    'detect_chinese_support',
    'configure_matplotlib_font',
    'CHINESE_SUPPORTED',
    'get_text',
    'simulate_training'
]
