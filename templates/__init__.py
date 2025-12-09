"""
Neural Network Architecture Templates

这个模块包含各种预定义的神经网络架构模板
每个模板都是一个独立的配置文件，易于维护和扩展
"""

from .template_loader import TemplateLoader, NetworkTemplate

__all__ = ["TemplateLoader", "NetworkTemplate"]
