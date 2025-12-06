"""
标签页模块
"""

from .params_calculator import params_calculator_tab
from .math_derivation import math_derivation_tab
from .failure_museum import failure_museum_tab
from .resnet_analysis import resnet_analysis_tab
from .normalization_comparison import normalization_comparison_tab
from .vit_analysis import vit_analysis_tab
from .architecture_comparison import architecture_comparison_tab
from .memory_analysis import memory_analysis_tab

__all__ = [
    'params_calculator_tab',
    'math_derivation_tab',
    'failure_museum_tab',
    'resnet_analysis_tab',
    'normalization_comparison_tab',
    'vit_analysis_tab',
    'architecture_comparison_tab',
    'memory_analysis_tab',
]
