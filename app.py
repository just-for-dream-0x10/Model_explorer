"""
神经网络架构的计算解剖台
Neural Network Math Explorer - 主应用入口

专注于单个神经元和层级计算细节
"""

import streamlit as st

# 导入工具模块
from utils import CHINESE_SUPPORTED, get_text

# 导入标签页模块
from tabs.params_calculator import params_calculator_tab
from tabs.math_derivation import math_derivation_tab
from tabs.backpropagation import backpropagation_tab
from tabs.interactive_lab import interactive_lab_tab
from tabs.failure_museum import failure_museum_tab
from tabs.resnet_analysis import resnet_analysis_tab
from tabs.normalization_comparison import normalization_comparison_tab
from tabs.vit_analysis import vit_analysis_tab
from tabs.architecture_comparison import architecture_comparison_tab
from tabs.memory_analysis import memory_analysis_tab
from tabs.stability_diagnosis import stability_diagnosis_tab
from tabs.architecture_designer import architecture_designer_tab
from tabs.moe_analysis import moe_analysis_tab
from tabs.model_pruning import model_pruning_tab
from cnn import cnn_tab
from gnn import gnn_tab
from rnn_lstm import rnn_lstm_tab

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
    st.markdown("[GitHub](https://github.com)")
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
        }
    else:  # 🚀 现代架构
        module_options = {
            "🔍 Vision Transformer分析": "vit_analysis",
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
        }
    else:  # 🚀 Modern Architectures
        module_options = {
            "🔍 ViT Analysis": "vit_analysis",
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
# 根据选择的模块显示内容
# ==========================================
if selected_module == "params_calculator":
    params_calculator_tab()
elif selected_module == "memory_analysis":
    memory_analysis_tab(CHINESE_SUPPORTED)
elif selected_module == "stability_diagnosis":
    stability_diagnosis_tab(CHINESE_SUPPORTED)
elif selected_module == "architecture_designer":
    architecture_designer_tab(CHINESE_SUPPORTED)
elif selected_module == "math_derivation":
    math_derivation_tab()
elif selected_module == "interactive_lab":
    interactive_lab_tab(CHINESE_SUPPORTED)
elif selected_module == "cnn":
    cnn_tab(CHINESE_SUPPORTED)
elif selected_module == "gnn":
    gnn_tab(CHINESE_SUPPORTED)
elif selected_module == "rnn_lstm":
    rnn_lstm_tab(CHINESE_SUPPORTED)
elif selected_module == "backpropagation":
    backpropagation_tab(CHINESE_SUPPORTED)
elif selected_module == "failure_museum":
    failure_museum_tab(CHINESE_SUPPORTED)
elif selected_module == "resnet_analysis":
    resnet_analysis_tab(CHINESE_SUPPORTED)
elif selected_module == "normalization":
    normalization_comparison_tab(CHINESE_SUPPORTED)
elif selected_module == "vit_analysis":
    vit_analysis_tab(CHINESE_SUPPORTED)
elif selected_module == "architecture_comparison":
    architecture_comparison_tab(selected_module)
elif selected_module == "moe_analysis":
    moe_analysis_tab(CHINESE_SUPPORTED)
elif selected_module == "model_pruning":
    model_pruning_tab(CHINESE_SUPPORTED)

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
