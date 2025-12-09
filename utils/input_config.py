"""
统一的输入配置组件
Unified Input Configuration Component

提供一致的输入形状配置界面，支持预设和自定义
"""

import streamlit as st
from typing import Tuple, Optional, Dict, Any


# 预设配置
PRESET_CONFIGS = {
    "MNIST (28×28)": {
        "shape": (1, 1, 28, 28),
        "description": "手写数字数据集，单通道灰度图",
        "icon": "📱",
        "typical_use": "入门学习、快速原型",
    },
    "CIFAR-10 (32×32)": {
        "shape": (1, 3, 32, 32),
        "description": "10类物体分类，RGB彩色图",
        "icon": "🖼️",
        "typical_use": "图像分类、数据增强实验",
    },
    "ImageNet (224×224)": {
        "shape": (1, 3, 224, 224),
        "description": "大规模图像识别，RGB彩色图",
        "icon": "🏞️",
        "typical_use": "迁移学习、预训练模型",
    },
    "高清 (512×512)": {
        "shape": (1, 3, 512, 512),
        "description": "高分辨率图像处理",
        "icon": "🎨",
        "typical_use": "图像分割、目标检测",
    },
    "自定义": {
        "shape": None,
        "description": "自定义输入形状",
        "icon": "⚙️",
        "typical_use": "特殊需求、实验探索",
    },
}


def render_input_config(
    default_preset: str = "ImageNet (224×224)",
    key_prefix: str = "input_config",
    show_batch_size: bool = False,
    show_description: bool = True,
    allow_custom: bool = True,
) -> Tuple[int, int, int, int]:
    """
    渲染输入配置组件

    参数:
        default_preset: 默认预设配置名称
        key_prefix: Streamlit 组件的 key 前缀
        show_batch_size: 是否显示批大小配置
        show_description: 是否显示配置描述
        allow_custom: 是否允许自定义配置

    返回:
        (batch_size, channels, height, width) 元组
    """

    # 准备预设列表
    preset_options = list(PRESET_CONFIGS.keys())
    if not allow_custom:
        preset_options = [p for p in preset_options if p != "自定义"]

    # 查找默认预设的索引
    try:
        default_index = preset_options.index(default_preset)
    except ValueError:
        default_index = 0

    # 预设选择
    col1, col2 = st.columns([2, 3])

    with col1:
        preset = st.selectbox(
            "📐 输入配置预设",
            preset_options,
            index=default_index,
            key=f"{key_prefix}_preset",
            help="选择常用的输入配置或自定义",
        )

    config = PRESET_CONFIGS[preset]

    # 显示描述
    if show_description and preset != "自定义":
        with col2:
            st.info(f"{config['icon']} {config['description']}")

    # 根据选择渲染配置
    if preset == "自定义":
        return _render_custom_config(key_prefix, show_batch_size)
    else:
        batch_size, channels, height, width = config["shape"]

        # 可选：允许微调批大小
        if show_batch_size:
            batch_size = st.number_input(
                "批大小 (Batch Size)",
                min_value=1,
                max_value=256,
                value=batch_size,
                key=f"{key_prefix}_batch_size",
                help="同时处理的样本数量",
            )

        # 显示完整配置信息
        _show_config_summary(batch_size, channels, height, width)

        return (batch_size, channels, height, width)


def _render_custom_config(
    key_prefix: str, show_batch_size: bool
) -> Tuple[int, int, int, int]:
    """渲染自定义配置界面"""

    st.markdown("#### ⚙️ 自定义输入配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        if show_batch_size:
            batch_size = st.number_input(
                "批大小",
                min_value=1,
                max_value=256,
                value=1,
                key=f"{key_prefix}_custom_batch",
            )
        else:
            batch_size = 1

    with col2:
        channels = st.selectbox(
            "通道数",
            [1, 3, 4],
            index=1,
            key=f"{key_prefix}_custom_channels",
            help="1=灰度图, 3=RGB彩色图, 4=RGBA",
        )

    with col3:
        img_size = st.number_input(
            "图像尺寸",
            min_value=8,
            max_value=1024,
            value=224,
            step=8,
            key=f"{key_prefix}_custom_size",
            help="假设为正方形图像 (H=W)",
        )

    # 高级选项：非正方形图像
    with st.expander("🔧 高级选项", expanded=False):
        use_rectangle = st.checkbox(
            "使用非正方形图像", key=f"{key_prefix}_rectangle", help="允许高度和宽度不同"
        )

        if use_rectangle:
            col_h, col_w = st.columns(2)
            with col_h:
                height = st.number_input(
                    "高度 (Height)",
                    min_value=8,
                    max_value=1024,
                    value=img_size,
                    step=8,
                    key=f"{key_prefix}_height",
                )
            with col_w:
                width = st.number_input(
                    "宽度 (Width)",
                    min_value=8,
                    max_value=1024,
                    value=img_size,
                    step=8,
                    key=f"{key_prefix}_width",
                )
        else:
            height = width = img_size

    # 显示配置摘要
    _show_config_summary(batch_size, channels, height, width)

    return (batch_size, channels, height, width)


def _show_config_summary(batch_size: int, channels: int, height: int, width: int):
    """显示配置摘要"""

    # 计算内存占用（假设 float32）
    memory_mb = batch_size * channels * height * width * 4 / (1024**2)
    total_pixels = channels * height * width

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "输入形状",
            f"({batch_size}, {channels}, {height}, {width})",
            help="(Batch, Channels, Height, Width)",
        )

    with col2:
        st.metric(
            "总像素数", f"{total_pixels:,}", help=f"{channels} × {height} × {width}"
        )

    with col3:
        st.metric(
            "内存占用", f"{memory_mb:.2f} MB", help="单个批次的内存使用（float32）"
        )


def get_preset_shape(preset_name: str) -> Optional[Tuple[int, int, int, int]]:
    """
    获取预设配置的形状

    参数:
        preset_name: 预设名称

    返回:
        (batch_size, channels, height, width) 或 None
    """
    config = PRESET_CONFIGS.get(preset_name)
    return config["shape"] if config else None


def calculate_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    计算卷积/池化层的输出尺寸

    公式: output = floor((input + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1

    参数:
        input_size: 输入尺寸
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率

    返回:
        输出尺寸
    """
    numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    output_size = numerator // stride + 1
    return output_size


def calculate_conv_output_shape(
    input_shape: Tuple[int, int, int, int],
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """
    计算卷积层的输出形状

    参数:
        input_shape: 输入形状 (B, C, H, W)
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充

    返回:
        输出形状 (B, C_out, H_out, W_out)
    """
    batch_size, in_channels, height, width = input_shape

    h_out = calculate_output_size(height, kernel_size, stride, padding)
    w_out = calculate_output_size(width, kernel_size, stride, padding)

    return (batch_size, out_channels, h_out, w_out)


def calculate_pool_output_shape(
    input_shape: Tuple[int, int, int, int],
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """
    计算池化层的输出形状

    参数:
        input_shape: 输入形状 (B, C, H, W)
        kernel_size: 池化核大小
        stride: 步长（默认等于 kernel_size）
        padding: 填充

    返回:
        输出形状 (B, C, H_out, W_out)
    """
    batch_size, channels, height, width = input_shape

    if stride is None:
        stride = kernel_size

    h_out = calculate_output_size(height, kernel_size, stride, padding)
    w_out = calculate_output_size(width, kernel_size, stride, padding)

    return (batch_size, channels, h_out, w_out)


def render_shape_flow_diagram(shapes: list, layer_names: list):
    """
    渲染形状流动图

    参数:
        shapes: 形状列表
        layer_names: 层名称列表
    """
    st.markdown("#### 🔄 形状变化流程")

    flow_text = ""
    for i, (shape, name) in enumerate(zip(shapes, layer_names)):
        b, c, h, w = shape
        flow_text += f"**{name}**: `({b}, {c}, {h}, {w})`"

        if i < len(shapes) - 1:
            flow_text += " → "

        # 每3个换行
        if (i + 1) % 3 == 0 and i < len(shapes) - 1:
            flow_text += "\n\n"

    st.markdown(flow_text)


# 使用示例
if __name__ == "__main__":
    st.set_page_config(page_title="输入配置组件测试", layout="wide")

    st.title("🎯 输入配置组件测试")

    st.markdown("---")

    # 示例 1: 基础使用
    st.markdown("## 示例 1: 基础使用")
    input_shape = render_input_config()
    st.success(f"选择的输入形状: {input_shape}")

    st.markdown("---")

    # 示例 2: 带批大小
    st.markdown("## 示例 2: 显示批大小配置")
    input_shape_2 = render_input_config(
        default_preset="MNIST (28×28)", key_prefix="example2", show_batch_size=True
    )
    st.success(f"选择的输入形状: {input_shape_2}")

    st.markdown("---")

    # 示例 3: 计算卷积输出
    st.markdown("## 示例 3: 计算卷积层输出")

    col1, col2, col3 = st.columns(3)
    with col1:
        kernel_size = st.slider("卷积核大小", 1, 7, 3, key="conv_k")
    with col2:
        stride = st.slider("步长", 1, 4, 1, key="conv_s")
    with col3:
        padding = st.slider("填充", 0, 3, 1, key="conv_p")

    out_channels = st.number_input("输出通道数", 1, 512, 64, key="conv_out")

    output_shape = calculate_conv_output_shape(
        input_shape, out_channels, kernel_size, stride, padding
    )

    st.info(f"卷积层输出形状: {output_shape}")

    # 显示形状流动
    render_shape_flow_diagram([input_shape, output_shape], ["输入", "Conv2d"])
