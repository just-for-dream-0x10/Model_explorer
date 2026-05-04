"""
神经网络架构的计算解剖台
Neural Network Math Explorer - 主应用入口

专注于单个神经元和层级计算细节
"""

import streamlit as st
import importlib

# 导入工具模块
from utils import CHINESE_SUPPORTED, get_text

# 标签页模块懒加载映射：模块名 -> (模块路径, 函数名)
_TAB_MODULES = {
    "params_calculator": ("tabs.params_calculator", "params_calculator_tab"),
    "math_derivation": ("tabs.math_derivation", "math_derivation_tab"),
    "memory_analysis": ("tabs.memory_analysis", "memory_analysis_tab"),
    "stability_diagnosis": ("tabs.stability_diagnosis", "stability_diagnosis_tab"),
    "architecture_designer": ("tabs.architecture_designer", "architecture_designer_tab"),
    "single_neuron": ("tabs.single_neuron", "single_neuron_tab"),
    "backpropagation": ("tabs.backpropagation", "backpropagation_tab"),
    "interactive_lab": ("tabs.interactive_lab", "interactive_lab_tab"),
    "failure_museum": ("tabs.failure_museum", "failure_museum_tab"),
    "resnet_analysis": ("tabs.resnet_analysis", "resnet_analysis_tab"),
    "normalization_comparison": ("tabs.normalization_comparison", "normalization_comparison_tab"),
    "vit_analysis": ("tabs.vit_analysis", "vit_analysis_tab"),
    "architecture_comparison": ("tabs.architecture_comparison", "architecture_comparison_tab"),
    "moe_analysis": ("tabs.moe_analysis", "moe_analysis_tab"),
    "model_pruning": ("tabs.model_pruning", "model_pruning_tab"),
    "performance_monitor": ("tabs.performance_monitor", "performance_monitor_tab"),
    "attention_analysis": ("tabs.attention_analysis", "attention_analysis_tab"),
    "model_compression": ("tabs.model_compression", "model_compression_tab"),
    "cnn": ("cnn", "cnn_tab"),
    "gnn": ("gnn", "gnn_tab"),
    "rnn_lstm": ("rnn_lstm", "rnn_lstm_tab"),
}

# 模块缓存，避免重复导入
_tab_cache = {}


def _load_tab(module_name):
    """按需加载标签页模块"""
    if module_name not in _tab_cache:
        module_path, func_name = _TAB_MODULES[module_name]
        mod = importlib.import_module(module_path)
        _tab_cache[module_name] = getattr(mod, func_name)
    return _tab_cache[module_name]

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="Neural Network Math Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 主标题和介绍
# ==========================================
st.title("🔬 神经网络架构的计算解剖台")

# ==========================================
# 侧边栏参数控制
# ==========================================
with st.sidebar:
    st.header(get_text("params_title"))

    st.subheader("🎛️ 全局参数")
    learning_rate = st.slider(
        "学习率" if CHINESE_SUPPORTED else "Learning Rate",
        0.0001,
        0.1,
        0.001,
        format="%.4f",
    )
    batch_size = st.slider(
        "批次大小" if CHINESE_SUPPORTED else "Batch Size", 8, 128, 32
    )

    st.markdown("---")
    st.markdown("### 📚 项目信息")
    st.markdown("**开发者**: Just For Dream Lab")
    st.markdown("[GitHub](https://github.com/just-for-dream-0x10)")
    st.markdown("[文档](./README.md)")

# ==========================================
# 侧边栏导航
# ==========================================
st.sidebar.title("📚 模块导航" if CHINESE_SUPPORTED else "📚 Module Navigation")

# 分类选择
category = st.sidebar.radio(
    "选择分类" if CHINESE_SUPPORTED else "Select Category",
    [
        "🔧 基础工具" if CHINESE_SUPPORTED else "🔧 Basic Tools",
        "🏗️ 经典架构" if CHINESE_SUPPORTED else "🏗️ Classic Architectures",
        "🎯 深度优化" if CHINESE_SUPPORTED else "🎯 Deep Optimization",
        "🚀 现代架构" if CHINESE_SUPPORTED else "🚀 Modern Architectures",
    ],
)

# 根据分类显示模块列表
if CHINESE_SUPPORTED:
    if category == "🔧 基础工具":
        module_options = {
            "🔢 参数量计算器": "params_calculator",
            "💾 内存分析器": "memory_analysis",
            "⚠️ 数值稳定性诊断": "stability_diagnosis",
            "🎨 架构设计工作台": "architecture_designer",
            "📐 数学推导工具": "math_derivation",
            "🎮 交互实验室": "interactive_lab",
            "🧬 单神经元分析": "single_neuron",
        }
    elif category == "🏗️ 经典架构":
        module_options = {
            "🖼️ CNN卷积数学": "cnn",
            "🕸️ GNN图神经网络": "gnn",
            "🔁 RNN/LSTM时序网络": "rnn_lstm",
            "🔬 反向传播原理": "backpropagation",
        }
    elif category == "🎯 深度优化":
        module_options = {
            "🏛️ 失败案例博物馆": "failure_museum",
            "🏗️ ResNet残差分析": "resnet_analysis",
            "🔧 归一化层对比": "normalization",
            "📦 模型压缩分析": "model_compression",
        }
    else:  # 🚀 现代架构
        module_options = {
            "🔍 Vision Transformer分析": "vit_analysis",
            "🧠 注意力机制分析": "attention_analysis",
            "🔬 架构对比实验室": "architecture_comparison",
            "🧠 MoE专家混合分析": "moe_analysis",
            "✂️ 模型剪枝分析": "model_pruning",
        }
else:
    if category == "🔧 Basic Tools":
        module_options = {
            "🔢 Params Calculator": "params_calculator",
            "💾 Memory Analyzer": "memory_analysis",
            "⚠️ Stability Diagnosis": "stability_diagnosis",
            "🎨 Architecture Designer": "architecture_designer",
            "📐 Math Derivation": "math_derivation",
            "🎮 Interactive Lab": "interactive_lab",
            "🧬 Single Neuron": "single_neuron",
        }
    elif category == "🏗️ Classic Architectures":
        module_options = {
            "🖼️ CNN": "cnn",
            "🕸️ GNN": "gnn",
            "🔁 RNN/LSTM": "rnn_lstm",
            "🔬 Backpropagation": "backpropagation",
        }
    elif category == "🎯 Deep Optimization":
        module_options = {
            "🏛️ Failure Museum": "failure_museum",
            "🏗️ ResNet Analysis": "resnet_analysis",
            "🔧 Normalization": "normalization",
            "📦 Model Compression": "model_compression",
        }
    else:  # 🚀 Modern Architectures
        module_options = {
            "🔍 ViT Analysis": "vit_analysis",
            "🧠 Attention Analysis": "attention_analysis",
            "🔬 Architecture Lab": "architecture_comparison",
            "🧠 MoE Analysis": "moe_analysis",
            "✂️ Model Pruning": "model_pruning",
        }

# 模块选择
selected_module_name = st.sidebar.selectbox(
    "选择模块" if CHINESE_SUPPORTED else "Select Module", list(module_options.keys())
)

selected_module = module_options[selected_module_name]

# 显示分隔线
st.sidebar.markdown("---")

# 显示当前模块信息
st.sidebar.info(
    f"📍 当前模块：{selected_module_name}"
    if CHINESE_SUPPORTED
    else f"📍 Current: {selected_module_name}"
)

# ==========================================
# 根据选择的模块显示内容（懒加载）
# ==========================================
if selected_module in _TAB_MODULES:
    tab_func = _load_tab(selected_module)
    # 根据各模块的函数签名传递参数
    if selected_module == "params_calculator":
        tab_func()
    elif selected_module == "math_derivation":
        tab_func()
    elif selected_module == "architecture_comparison":
        tab_func(selected_module)
    else:
        tab_func(CHINESE_SUPPORTED)

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Neural Network Math Explorer v1.5.0</p>
    <p>专注于网络层计算细节 | Just For Dream Lab</p>
</div>
""",
    unsafe_allow_html=True,
)
