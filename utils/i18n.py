"""
国际化文本配置
"""

from .config import CHINESE_SUPPORTED


# 中英文文本配置
TEXT_CONFIG = {
    "chinese": {
        "title": "神经网络数学原理探索器",
        "subtitle": "深入理解CNN、GNN、RNN等神经网络的核心数学原理",
        "description": "**交互式数学实验室** - 通过严谨的数学推导和可视化深入理解神经网络算法",
        "cnn_tab": "🔄 CNN卷积数学",
        "gnn_tab": "🕸️ GNN图神经网络",
        "rnn_tab": "🔄 RNN/LSTM时序网络",
        "math_tab": "📐 数学推导工具",
        "bp_tab": "🔬 反向传播原理",
        "lab_tab": "🎮 交互实验室",
        "params_title": "🎛️ 实验参数",
        "cnn_params": "CNN 参数",
        "gnn_params": "GNN 参数",
        "rnn_params": "RNN/LSTM 参数",
        "common_params": "通用参数",
        "learning_rate": "学习率",
    },
    "english": {
        "title": "Neural Network Mathematics Explorer",
        "subtitle": "Deep Understanding of Core Mathematical Principles in CNN, GNN, RNN and Other Neural Networks",
        "description": "**Interactive Mathematics Lab** - Deep understanding of neural network algorithms through rigorous mathematical derivations and visualizations",
        "cnn_tab": "🔄 CNN Convolution Math",
        "gnn_tab": "🕸️ GNN Graph Neural Networks",
        "rnn_tab": "🔄 RNN/LSTM Sequential Networks",
        "math_tab": "📐 Math Derivation Tools",
        "bp_tab": "🔬 Backpropagation Principles",
        "lab_tab": "🎮 Interactive Lab",
        "params_title": "🎛️ Experiment Parameters",
        "cnn_params": "CNN Parameters",
        "gnn_params": "GNN Parameters",
        "rnn_params": "RNN/LSTM Parameters",
        "common_params": "Common Parameters",
        "learning_rate": "Learning Rate",
    },
}


def get_text(key):
    """根据系统语言获取对应文本"""
    language = "chinese" if CHINESE_SUPPORTED else "english"
    return TEXT_CONFIG[language].get(key, key)
