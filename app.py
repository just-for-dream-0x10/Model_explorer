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
    initial_sidebar_state="expanded"
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
        0.0001, 0.1, 0.001, 
        format="%.4f"
    )
    batch_size = st.slider(
        "批次大小" if CHINESE_SUPPORTED else "Batch Size", 
        8, 128, 32
    )
    
    st.markdown("---")
    st.markdown("### 📚 项目信息")
    st.markdown("**开发者**: Just For Dream Lab")
    st.markdown("[GitHub](https://github.com)")
    st.markdown("[文档](./README.md)")

# ==========================================
# 标签页
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🔢 参数量计算器" if CHINESE_SUPPORTED else "🔢 Params Calculator",
    get_text("cnn_tab"),
    get_text("gnn_tab"),
    get_text("rnn_tab"),
    get_text("math_tab"),
    get_text("bp_tab"),
    "🎮 交互实验室" if CHINESE_SUPPORTED else "🎮 Interactive Lab",
    "🏛️ 失败案例博物馆" if CHINESE_SUPPORTED else "🏛️ Failure Museum",
    "🏗️ ResNet残差分析" if CHINESE_SUPPORTED else "🏗️ ResNet Analysis",
    "🔧 归一化层对比" if CHINESE_SUPPORTED else "🔧 Normalization",
])

# TAB 1: 参数量计算器 (核心差异化功能)
with tab1:
    params_calculator_tab()

# TAB 2: CNN卷积数学
with tab2:
    cnn_tab(CHINESE_SUPPORTED)

# TAB 3: GNN图神经网络
with tab3:
    gnn_tab(CHINESE_SUPPORTED)

# TAB 4: RNN/LSTM时序网络
with tab4:
    rnn_lstm_tab(CHINESE_SUPPORTED)

# TAB 5: 数学推导工具
with tab5:
    math_derivation_tab()

# TAB 6: 反向传播原理
with tab6:
    backpropagation_tab(CHINESE_SUPPORTED)

# TAB 7: 交互实验室
with tab7:
    interactive_lab_tab(CHINESE_SUPPORTED)

# TAB 8: 失败案例博物馆
with tab8:
    failure_museum_tab(CHINESE_SUPPORTED)

# TAB 9: ResNet残差分析
with tab9:
    resnet_analysis_tab(CHINESE_SUPPORTED)

# TAB 10: 归一化层对比
with tab10:
    normalization_comparison_tab(CHINESE_SUPPORTED)

# ==========================================
# 页脚
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Neural Network Math Explorer v1.5.0</p>
    <p>专注于网络层计算细节 | Just For Dream Lab</p>
</div>
""", unsafe_allow_html=True)
