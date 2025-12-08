"""
层参数配置组件
Layer Parameters Configuration Component

提供统一的层参数配置界面
"""

import streamlit as st
from typing import Dict, Any, Optional, List


def render_conv2d_params(
    key_prefix: str = "conv",
    default_kernel_size: int = 3,
    default_stride: int = 1,
    default_padding: int = 1,
    show_advanced: bool = True,
) -> Dict[str, Any]:
    """
    渲染 Conv2d 层参数配置

    参数:
        key_prefix: 组件 key 前缀
        default_kernel_size: 默认卷积核大小
        default_stride: 默认步长
        default_padding: 默认填充
        show_advanced: 是否显示高级选项

    返回:
        参数字典
    """
    st.markdown("#### 🔧 卷积层参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        kernel_size = st.slider(
            "卷积核大小",
            min_value=1,
            max_value=7,
            value=default_kernel_size,
            step=2,  # 通常使用奇数
            key=f"{key_prefix}_kernel",
            help="卷积核的空间尺寸 (通常使用奇数，如3×3, 5×5)",
        )

    with col2:
        stride = st.slider(
            "步长 (Stride)",
            min_value=1,
            max_value=4,
            value=default_stride,
            key=f"{key_prefix}_stride",
            help="卷积核移动的步长，越大输出越小",
        )

    with col3:
        padding = st.slider(
            "填充 (Padding)",
            min_value=0,
            max_value=3,
            value=default_padding,
            key=f"{key_prefix}_padding",
            help="输入周围添加的零填充层数",
        )

    params = {"kernel_size": kernel_size, "stride": stride, "padding": padding}

    # 高级选项
    if show_advanced:
        with st.expander("🔬 高级选项", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                dilation = st.slider(
                    "膨胀率 (Dilation)",
                    min_value=1,
                    max_value=3,
                    value=1,
                    key=f"{key_prefix}_dilation",
                    help="卷积核元素之间的间距",
                )

            with col2:
                groups = st.selectbox(
                    "分组 (Groups)",
                    [1, 2, 4, 8],
                    index=0,
                    key=f"{key_prefix}_groups",
                    help="分组卷积，groups=1为标准卷积",
                )

            with col3:
                use_bias = st.checkbox(
                    "使用偏置",
                    value=True,
                    key=f"{key_prefix}_bias",
                    help="是否在卷积后添加偏置项",
                )

            params.update({"dilation": dilation, "groups": groups, "bias": use_bias})

    # 显示参数摘要
    _show_conv_summary(params)

    return params


def render_pool_params(
    key_prefix: str = "pool", pool_type: str = "MaxPool2d", default_kernel_size: int = 2
) -> Dict[str, Any]:
    """
    渲染池化层参数配置

    参数:
        key_prefix: 组件 key 前缀
        pool_type: 池化类型 ("MaxPool2d" 或 "AvgPool2d")
        default_kernel_size: 默认池化核大小

    返回:
        参数字典
    """
    st.markdown(f"#### 🔧 {pool_type} 参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        kernel_size = st.selectbox(
            "池化核大小",
            [2, 3, 4],
            index=(
                [2, 3, 4].index(default_kernel_size)
                if default_kernel_size in [2, 3, 4]
                else 0
            ),
            key=f"{key_prefix}_kernel",
            help="池化窗口的大小",
        )

    with col2:
        stride = st.selectbox(
            "步长",
            [None, 1, 2, 3, 4],
            index=0,
            format_func=lambda x: "等于 kernel_size" if x is None else str(x),
            key=f"{key_prefix}_stride",
            help="None 表示步长等于 kernel_size",
        )

    with col3:
        padding = st.slider(
            "填充",
            min_value=0,
            max_value=2,
            value=0,
            key=f"{key_prefix}_padding",
            help="池化前的填充",
        )

    params = {
        "kernel_size": kernel_size,
        "stride": stride if stride is not None else kernel_size,
        "padding": padding,
    }

    return params


def render_linear_params(
    key_prefix: str = "linear",
    default_out_features: int = 128,
    min_features: int = 10,
    max_features: int = 2048,
) -> Dict[str, Any]:
    """
    渲染全连接层参数配置

    参数:
        key_prefix: 组件 key 前缀
        default_out_features: 默认输出特征数
        min_features: 最小特征数
        max_features: 最大特征数

    返回:
        参数字典
    """
    st.markdown("#### 🔧 全连接层参数")

    col1, col2 = st.columns(2)

    with col1:
        out_features = st.number_input(
            "输出特征数",
            min_value=min_features,
            max_value=max_features,
            value=default_out_features,
            step=64,
            key=f"{key_prefix}_out_features",
            help="全连接层的输出维度",
        )

    with col2:
        use_bias = st.checkbox(
            "使用偏置", value=True, key=f"{key_prefix}_bias", help="是否添加偏置项"
        )

    params = {"out_features": out_features, "bias": use_bias}

    return params


def render_activation_selector(
    key_prefix: str = "activation", default: str = "ReLU"
) -> Dict[str, Any]:
    """
    渲染激活函数选择器

    参数:
        key_prefix: 组件 key 前缀
        default: 默认激活函数

    返回:
        包含激活函数类型和参数的字典
    """
    st.markdown("#### ⚡ 激活函数")

    col1, col2 = st.columns([2, 3])

    with col1:
        activation_type = st.selectbox(
            "激活函数类型",
            ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "ELU"],
            index=(
                ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "ELU"].index(default)
                if default in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "ELU"]
                else 0
            ),
            key=f"{key_prefix}_type",
            help="选择非线性激活函数",
        )

    params = {"type": activation_type}

    # 根据激活函数类型显示特定参数
    with col2:
        if activation_type == "LeakyReLU":
            negative_slope = st.slider(
                "负斜率",
                min_value=0.01,
                max_value=0.5,
                value=0.01,
                step=0.01,
                key=f"{key_prefix}_negative_slope",
                help="负半轴的斜率",
            )
            params["negative_slope"] = negative_slope

        elif activation_type == "ELU":
            alpha = st.slider(
                "Alpha",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key=f"{key_prefix}_alpha",
                help="ELU 的 alpha 参数",
            )
            params["alpha"] = alpha

    # 显示激活函数特性
    _show_activation_info(activation_type)

    return params


def render_dropout_params(
    key_prefix: str = "dropout", default_p: float = 0.5
) -> Dict[str, Any]:
    """
    渲染 Dropout 参数配置

    参数:
        key_prefix: 组件 key 前缀
        default_p: 默认丢弃率

    返回:
        参数字典
    """
    st.markdown("#### 🎲 Dropout 参数")

    p = st.slider(
        "丢弃率 (p)",
        min_value=0.0,
        max_value=0.9,
        value=default_p,
        step=0.05,
        key=f"{key_prefix}_p",
        help="训练时随机丢弃神经元的比例",
    )

    params = {"p": p}

    # 显示效果说明
    col1, col2 = st.columns(2)
    with col1:
        st.metric("保留比例", f"{(1-p)*100:.0f}%")
    with col2:
        st.metric("丢弃比例", f"{p*100:.0f}%")

    return params


def render_batchnorm_params(
    key_prefix: str = "batchnorm", show_advanced: bool = False
) -> Dict[str, Any]:
    """
    渲染 BatchNorm 参数配置

    参数:
        key_prefix: 组件 key 前缀
        show_advanced: 是否显示高级参数

    返回:
        参数字典
    """
    st.markdown("#### 📊 BatchNorm 参数")

    params = {}

    if show_advanced:
        col1, col2 = st.columns(2)

        with col1:
            momentum = st.slider(
                "Momentum",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key=f"{key_prefix}_momentum",
                help="用于运行平均值和方差的动量",
            )

        with col2:
            eps = st.number_input(
                "Epsilon",
                min_value=1e-6,
                max_value=1e-3,
                value=1e-5,
                format="%.2e",
                key=f"{key_prefix}_eps",
                help="添加到分母以提高数值稳定性",
            )

        params = {"momentum": momentum, "eps": eps}
    else:
        st.info("💡 BatchNorm 使用默认参数 (momentum=0.1, eps=1e-5)")

    return params


# ==================== 辅助函数 ====================


def _show_conv_summary(params: Dict[str, Any]):
    """显示卷积参数摘要"""
    col1, col2, col3 = st.columns(3)

    with col1:
        receptive_field = params["kernel_size"] + (params["kernel_size"] - 1) * (
            params.get("dilation", 1) - 1
        )
        st.metric("感受野", f"{receptive_field}×{receptive_field}")

    with col2:
        if (
            params["stride"] == 1
            and params["padding"] == (params["kernel_size"] - 1) // 2
        ):
            st.metric("输出尺寸", "保持不变", help="same padding")
        else:
            st.metric("输出尺寸", "会改变", help="输出尺寸 ≠ 输入尺寸")

    with col3:
        if params.get("groups", 1) > 1:
            st.metric("卷积类型", f"分组卷积 (×{params['groups']})")
        else:
            st.metric("卷积类型", "标准卷积")


def _show_activation_info(activation_type: str):
    """显示激活函数信息"""
    info = {
        "ReLU": {
            "range": "[0, +∞)",
            "pros": "计算简单、缓解梯度消失",
            "cons": "可能导致神经元死亡",
        },
        "LeakyReLU": {
            "range": "(-∞, +∞)",
            "pros": "解决 ReLU 死神经元问题",
            "cons": "负斜率需要调参",
        },
        "Sigmoid": {"range": "(0, 1)", "pros": "输出概率解释", "cons": "梯度消失严重"},
        "Tanh": {"range": "(-1, 1)", "pros": "零中心化", "cons": "梯度消失问题"},
        "GELU": {
            "range": "(-∞, +∞)",
            "pros": "Transformer中常用",
            "cons": "计算稍复杂",
        },
        "ELU": {"range": "(-α, +∞)", "pros": "负值时更平滑", "cons": "计算涉及指数"},
    }

    if activation_type in info:
        details = info[activation_type]
        st.info(
            f"**值域**: {details['range']}\n\n"
            f"**优点**: {details['pros']}\n\n"
            f"**缺点**: {details['cons']}"
        )


def render_layer_params_sidebar():
    """
    在侧边栏渲染通用层参数配置

    返回:
        包含所有参数的字典
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ 层参数配置")
    st.sidebar.markdown("调整这些参数会影响所有示例")

    params = {}

    with st.sidebar.expander("🔲 卷积层", expanded=False):
        params["conv"] = render_conv2d_params(
            key_prefix="sidebar_conv", show_advanced=False
        )

    with st.sidebar.expander("⬇️ 池化层", expanded=False):
        params["pool"] = render_pool_params(key_prefix="sidebar_pool")

    with st.sidebar.expander("⚡ 激活函数", expanded=False):
        params["activation"] = render_activation_selector(
            key_prefix="sidebar_activation"
        )

    with st.sidebar.expander("🎲 Dropout", expanded=False):
        params["dropout"] = render_dropout_params(key_prefix="sidebar_dropout")

    return params


# 使用示例
if __name__ == "__main__":
    st.set_page_config(page_title="层参数配置组件测试", layout="wide")

    st.title("🎛️ 层参数配置组件测试")

    tab1, tab2, tab3, tab4 = st.tabs(["卷积层", "池化层", "激活函数", "其他"])

    with tab1:
        st.markdown("## 卷积层参数")
        conv_params = render_conv2d_params(key_prefix="test_conv")
        st.json(conv_params)

    with tab2:
        st.markdown("## 池化层参数")
        pool_params = render_pool_params(key_prefix="test_pool")
        st.json(pool_params)

    with tab3:
        st.markdown("## 激活函数")
        activation_params = render_activation_selector(key_prefix="test_activation")
        st.json(activation_params)

    with tab4:
        st.markdown("## Dropout")
        dropout_params = render_dropout_params(key_prefix="test_dropout")
        st.json(dropout_params)

        st.markdown("## BatchNorm")
        bn_params = render_batchnorm_params(key_prefix="test_bn", show_advanced=True)
        st.json(bn_params)
