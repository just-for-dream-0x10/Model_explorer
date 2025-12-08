"""
CNN卷积神经网络数学原理模块
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from simple_latex import display_latex

from utils.visualization import ChartBuilder, MathVisualization
from utils.input_config import (
    render_input_config,
    calculate_conv_output_shape,
    calculate_output_size,
)
from utils.layer_params import render_conv2d_params, render_activation_selector


# 辅助函数：生成不同类型的图案
def create_checkerboard(size, square_size=8):
    """创建棋盘格图案"""
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i // square_size + j // square_size) % 2 == 0:
                pattern[i, j] = 1
    return pattern


def create_concentric_circles(size, center=None, rings=5):
    """创建同心圆图案"""
    if center is None:
        center = (size // 2, size // 2)

    pattern = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]

    max_radius = np.sqrt(2) * size / 2
    for i in range(rings):
        radius = (i + 1) * max_radius / rings
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        if i % 2 == 0:
            pattern[mask] = 1

    return pattern


def create_gradient(size, direction="diagonal"):
    """创建渐变图案"""
    pattern = np.zeros((size, size))

    if direction == "horizontal":
        pattern = np.linspace(0, 1, size).reshape(1, -1).repeat(size, axis=0)
    elif direction == "vertical":
        pattern = np.linspace(0, 1, size).reshape(-1, 1).repeat(size, axis=1)
    elif direction == "diagonal":
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        pattern = (x + y) / 2
    elif direction == "radial":
        center = size // 2
        y, x = np.ogrid[:size, :size]
        pattern = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        pattern = pattern / pattern.max()

    return pattern


def cnn_tab(CHINESE_SUPPORTED):
    """CNN标签页内容"""

    st.header("🔄 CNN卷积操作数学原理")

    # 初始化图表工具
    chart_builder = ChartBuilder()
    math_viz = MathVisualization()

    # ==========================================
    # 输入和层参数配置（新增）
    # ==========================================
    st.markdown("### ⚙️ 配置选项")
    tab1, tab2 = st.tabs(["📐 输入配置", "🔧 层参数"])
    
    with tab1:
        st.markdown("配置用于计算示例的输入形状")
        input_shape = render_input_config(
            default_preset="CIFAR-10 (32×32)",
            key_prefix="cnn_input",
            show_batch_size=False,
            show_description=True
        )
        batch_size, channels, img_height, img_width = input_shape
    
    with tab2:
        st.markdown("配置卷积层和激活函数参数")
        conv_params = render_conv2d_params(
            key_prefix="cnn_conv",
            default_kernel_size=3,
            default_stride=1,
            default_padding=1,
            show_advanced=False  # 改为 False，避免嵌套 expander
        )
        activation_params = render_activation_selector(
            key_prefix="cnn_activation",
            default="ReLU"
        )

    # ==========================================
    # 第一部分：核心概念与直观理解
    # ==========================================
    st.markdown("### 🎯 卷积运算：像用放大镜看图片")

    with st.expander("💡 直观理解", expanded=True):
        st.markdown(
            """
        **卷积就像是用一个特殊的"放大镜"（卷积核）在图片上滑动：**
        
        1. 📍 **定位窗口** - 在图片上放置卷积核
        2. 🔢 **计算特征** - 窗口内像素与卷积核对应相乘再相加
        3. 📝 **记录结果** - 得到该位置的特征值
        4. 👉 **滑动窗口** - 移动到下一个位置重复
        
        **关键参数：**
        - **卷积核大小**：放大镜的大小（越大看范围越广）
        - **步长**：每次移动的距离（越大跳得越远）
        - **填充**：是否给图片加边框（让边缘也能看清）
        """
        )

    # ==========================================
    # 第二部分：实时演示与计算过程
    # ==========================================
    st.markdown("### 🔍 卷积过程实时演示")

    col1, col2 = st.columns([1, 1])

    with col1:
        # 卷积核类型选择
        kernel_types = {
            "边缘检测": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "模糊": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "浮雕": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        }

        selected_kernel_type = st.selectbox(
            "选择卷积核类型", list(kernel_types.keys()), key="kernel_type_select"
        )
        demo_kernel = kernel_types[selected_kernel_type]

        # 创建或上传图像
        st.markdown("**输入图像**")
        input_option = st.radio("输入方式", ["生成示例图案", "上传图像"])

        if input_option == "生成示例图案":
            pattern_types = {
                "棋盘格": create_checkerboard,
                "同心圆": create_concentric_circles,
                "随机噪声": lambda size: np.random.randn(size, size),
                "渐变": create_gradient,
            }
            selected_pattern = st.selectbox(
                "选择图案类型", list(pattern_types.keys()), key="pattern_type_select"
            )
            demo_size = st.slider("图案大小", 32, 128, 64)
            demo_input_image = pattern_types[selected_pattern](demo_size)
        else:
            uploaded_file = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                # 这里可以添加图像处理逻辑
                demo_input_image = np.random.randn(64, 64)  # 暂时用随机数据代替

        # 显示输入和卷积核
        fig_input = chart_builder.create_heatmap(
            demo_input_image, title="输入图像", colorscale="gray", height=250
        )
        chart_builder.display_chart(fig_input)

        st.markdown(f"**{selected_kernel_type}卷积核**")
        fig_kernel = chart_builder.create_heatmap(
            demo_kernel,
            title=f"{selected_kernel_type}检测器",
            colorscale="RdBu",
            height=200,
        )
        chart_builder.display_chart(fig_kernel)

    with col2:
        # 参数控制
        st.markdown("**🎛️ 参数控制**")
        demo_stride = st.slider("步长", 1, 4, 1)
        demo_padding = st.slider("填充", 0, 3, 0)

        # 执行卷积
        conv_result = signal.convolve2d(demo_input_image, demo_kernel, mode="same")

        # 应用步长
        if demo_stride > 1:
            conv_result = conv_result[::demo_stride, ::demo_stride]

        # 显示卷积结果
        st.markdown("**卷积结果**")
        fig_result = chart_builder.create_heatmap(
            conv_result, title="卷积输出", colorscale="viridis", height=250
        )
        chart_builder.display_chart(fig_result)

        # 显示具体计算示例
        if demo_input_image.shape[0] >= 3 and demo_input_image.shape[1] >= 3:
            st.markdown("**🧮 计算示例（位置0,0）**")
            demo_window = demo_input_image[0:3, 0:3]

            # 确保窗口和卷积核形状匹配
            if demo_window.shape == demo_kernel.shape:
                demo_conv_result = np.sum(demo_window * demo_kernel)
            else:
                min_shape = (
                    min(demo_window.shape[0], demo_kernel.shape[0]),
                    min(demo_window.shape[1], demo_kernel.shape[1]),
                )
                demo_conv_result = np.sum(
                    demo_window[: min_shape[0], : min_shape[1]]
                    * demo_kernel[: min_shape[0], : min_shape[1]]
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**输入窗口**")
                st.dataframe(pd.DataFrame(demo_window.round(2)).style.format("{:.2f}"))

            with col_b:
                st.markdown("**卷积核**")
                st.dataframe(pd.DataFrame(demo_kernel.round(2)).style.format("{:.2f}"))

            st.markdown(f"**结果**: {demo_conv_result:.3f}")

    # ==========================================
    # 第三部分：参数影响深度分析
    # ==========================================
    st.markdown("---")
    st.markdown("### 📊 参数影响深度分析")

    param_analysis = st.tabs(["卷积核大小", "步长影响", "填充策略"])

    with param_analysis[0]:
        st.markdown(
            """
        **🔍 卷积核大小的影响**
        
        | 特性 | 小卷积核(3×3) | 大卷积核(7×7) |
        |------|---------------|---------------|
        | 感受野 | 局部细节 | 全局特征 |
        | 参数量 | 少 | 多 |
        | 计算效率 | 高 | 低 |
        | 适用场景 | 细节检测 | 整体理解 |
        """
        )

        # 可视化不同核大小的影响
        input_demo = 32
        kernel_sizes = [3, 5, 7, 9]
        output_sizes = [(input_demo - ks) // 1 + 1 for ks in kernel_sizes]

        fig_kernel_size = chart_builder.create_line_chart(
            x_data=kernel_sizes,
            y_data=output_sizes,
            title="卷积核大小 vs 输出尺寸",
            x_title="卷积核大小",
            y_title="输出尺寸",
            height=300,
        )

        # 添加文本标签
        fig_kernel_size.update_traces(
            text=[f"{out}×{out}" for out in output_sizes], textposition="top center"
        )
        chart_builder.display_chart(fig_kernel_size)

    with param_analysis[1]:
        st.markdown(
            """
        **🏃 步长的权衡**
        
        **步长 = 1（精细扫描）**
        - ✅ 不遗漏任何信息
        - ❌ 计算量大
        - 🎯 适合需要高精度的任务
        
        **步长 > 1（快速扫描）**
        - ✅ 计算效率高
        - ❌ 可能丢失细节
        - 🎯 适合特征提取的下采样
        """
        )

        # 步长效率演示
        strides = [1, 2, 4, 8]
        kernel_demo = 3
        input_size_demo = 64

        efficiency_data = []
        for stride in strides:
            output_size = (input_size_demo - kernel_demo) // stride + 1
            speedup = (input_size_demo / stride) ** 2
            efficiency_data.append(
                {
                    "步长": stride,
                    "输出尺寸": f"{output_size}×{output_size}",
                    "计算量比例": f"1/{stride**2}",
                    "加速比": f"{speedup:.1f}x",
                }
            )

        df = pd.DataFrame(efficiency_data)
        st.dataframe(df)

    with param_analysis[2]:
        st.markdown(
            """
        **🎯 填充策略指南**
        
        **Same Padding（保持尺寸）**
        - 填充 = (卷积核大小 - 1) / 2
        - 输出尺寸 = 输入尺寸
        - 🎯 适合深层网络
        
        **Valid Padding（无填充）**
        - 填充 = 0
        - 输出尺寸 < 输入尺寸
        - 🎯 适合特征压缩
        
        **Full Padding（最大填充）**
        - 填充 = 卷积核大小 - 1
        - 输出尺寸 > 输入尺寸
        - 🎯 适合转置卷积
        """
        )

        # 填充效果可视化
        input_demo = 32
        kernel_demo = 5
        padding_options = [0, 1, 2, 3, 4]

        padding_effects = []
        for padding in padding_options:
            output_size = (input_demo - kernel_demo + 2 * padding) // 1 + 1
            edge_coverage = min(1.0, (kernel_demo // 2 + padding) / (input_demo / 2))
            padding_effects.append(
                {
                    "填充大小": padding,
                    "输出尺寸": output_size,
                    "边缘覆盖率": f"{edge_coverage:.1%}",
                    "策略": ["Valid", "Small", "Same", "Large", "Full"][padding],
                }
            )

        df = pd.DataFrame(padding_effects)
        st.dataframe(df)

    # ==========================================
    # 第四部分：数学公式与计算
    # ==========================================
    with st.expander("📐 数学公式推导（可选）"):
        col_formula, col_example = st.columns([1, 1])

        with col_formula:
            st.markdown("**卷积公式**")
            display_latex(r"(f * g)[i,j] = \sum_{m} \sum_{n} f[m,n] \cdot g[i-m, j-n]")

            st.markdown("**输出尺寸计算**")
            st.markdown(
                "$$H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - K}{S} \\right\\rfloor + 1$$"
            )
            st.markdown(
                "$$W_{out} = \\left\lfloor \\frac{W_{in} + 2P - K}{S} \\right\\rfloor + 1$$"
            )

            st.markdown("**参数说明**")
            st.markdown("- $H_{in}, W_{in}$: 输入高宽")
            st.markdown("- $H_{out}, W_{out}$: 输出高宽")
            st.markdown("- $K$: 卷积核大小")
            st.markdown("- $S$: 步长")
            st.markdown("- $P$: 填充大小")

        with col_example:
            st.markdown("**实际计算示例**")

            # 使用动态示例生成器
            from utils.example_generator import get_dynamic_example

            try:
                example = get_dynamic_example("cnn")

                st.markdown(
                    f"""
                **给定参数** (基于您的当前选择):
                - 输入尺寸: {example['input_size']} $ \\times $  {example['input_size']}
                - 卷积核: {example['kernel_size']} $ \\times $ {example['kernel_size']}
                - 步长: {example['stride']}
                - 填充: {example['padding']}
                
                **计算过程**:
                {example['calculation_formula']}
                
                **输出尺寸**: {example['output_size']} $ \\times $ {example['output_size']}
                """
                )
            except Exception as e:
                # 如果动态生成失败，使用用户配置的参数
                example_input_size = img_height
                example_kernel_size = conv_params['kernel_size']
                example_stride = conv_params['stride']
                example_padding = conv_params['padding']

                h_out = calculate_output_size(
                    example_input_size,
                    example_kernel_size,
                    example_stride,
                    example_padding,
                )

                st.markdown(
                    f"""
                **给定参数** (基于当前配置):
                - 输入尺寸: {example_input_size} $ \\times $  {example_input_size}
                - 卷积核: {example_kernel_size} $ \\times $ {example_kernel_size}
                - 步长: {example_stride}
                - 填充: {example_padding}
                
                **计算过程**:
                $ H_{{out}} = \\left\\lfloor \\frac{{H_{{in}} + 2P - K}}{{S}} \\right\\rfloor + 1 = \\frac{{{example_input_size} + 2 \\times {example_padding} - {example_kernel_size}}}{{{example_stride}}} + 1 = {h_out} $
                
                **输出尺寸**: {h_out} $ \\times $ {h_out}
                
                💡 **提示**: 在上方"⚙️ 配置选项"中可以调整所有参数
                """
                )
                
                # 显示参数影响
                st.info(f"""
                **参数影响分析**:
                - 卷积核越大 → 感受野越大，但计算量也越大
                - 步长越大 → 输出尺寸越小，下采样更激进
                - 填充越大 → 边界信息保留更多
                - 当前激活函数: {activation_params['type']}
                """)

    # ==========================================
    # 第五部分：手动计算演示
    # ==========================================
    st.markdown("---")
    st.markdown("### 🧮 手动计算演示")

    # 使用动态示例生成器
    from utils.example_generator import get_dynamic_example

    try:
        example = get_dynamic_example("cnn")
        kernel_size = example["kernel_size"]
        stride = example["stride"]
        padding = example["padding"]
        input_size = example["input_size"]
        input_matrix = example["input_matrix"]
        kernel = example["kernel"]
    except Exception as e:
        # 如果动态生成失败，使用默认参数
        kernel_size = 3
        stride = 1
        padding = 0
        input_size = 5
        input_matrix = np.random.randn(input_size, input_size).round(2)
        kernel = np.random.randn(kernel_size, kernel_size).round(2)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**输入矩阵**")
        st.dataframe(
            pd.DataFrame(input_matrix)
            .style.format("{:.2f}")
            .background_gradient(cmap="Blues")
        )

        st.markdown("**卷积核**")
        st.dataframe(
            pd.DataFrame(kernel).style.format("{:.2f}").background_gradient(cmap="Reds")
        )

    with col2:
        st.markdown("### 卷积计算过程")

        # 手动实现卷积计算
        output_size = (input_size - kernel_size + 2 * padding) // stride + 1
        output_matrix = np.zeros((output_size, output_size))

        # 创建逐步计算的可视化
        step_by_step = []
        for i in range(output_size):
            for j in range(output_size):
                # 提取当前窗口
                start_i = i * stride
                start_j = j * stride
                window = input_matrix[
                    start_i : start_i + kernel_size, start_j : start_j + kernel_size
                ]

                # 确保窗口和卷积核形状匹配
                if window.shape == kernel.shape:
                    # 计算卷积
                    conv_result = np.sum(window * kernel)
                else:
                    # 如果形状不匹配，使用有效区域或跳过
                    min_shape = (
                        min(window.shape[0], kernel.shape[0]),
                        min(window.shape[1], kernel.shape[1]),
                    )
                    conv_result = np.sum(
                        window[: min_shape[0], : min_shape[1]]
                        * kernel[: min_shape[0], : min_shape[1]]
                    )
                output_matrix[i, j] = conv_result

                step_by_step.append(
                    {
                        "position": f"({i},{j})",
                        "window": window.tolist(),
                        "kernel": kernel.tolist(),
                        "result": round(conv_result, 3),
                    }
                )

        st.markdown("### 输出结果")
        st.dataframe(
            pd.DataFrame(output_matrix)
            .style.format("{:.2f}")
            .background_gradient(cmap="Greens")
        )

        # 选择特定位置查看详细计算
        selected_pos = st.selectbox(
            "🔍 查看详细计算位置",
            [f"({i},{j})" for i in range(output_size) for j in range(output_size)],
            key="detail_calc_1",
        )

        for step in step_by_step:
            if step["position"] == selected_pos:
                st.markdown(f"#### 📍 位置 {selected_pos} 的详细计算")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**📋 输入窗口**")
                    st.dataframe(pd.DataFrame(step["window"]).style.format("{:.2f}"))

                with col_b:
                    st.markdown("**⚙️ 卷积核**")
                    st.dataframe(pd.DataFrame(step["kernel"]).style.format("{:.2f}"))

                with col_c:
                    st.markdown("**✖️ 逐元素乘积**")
                    window_arr = np.array(step["window"])
                    kernel_arr = np.array(step["kernel"])

                    # 确保形状匹配
                    if window_arr.shape == kernel_arr.shape:
                        element_product = window_arr * kernel_arr
                    else:
                        min_shape = (
                            min(window_arr.shape[0], kernel_arr.shape[0]),
                            min(window_arr.shape[1], kernel_arr.shape[1]),
                        )
                        element_product = (
                            window_arr[: min_shape[0], : min_shape[1]]
                            * kernel_arr[: min_shape[0], : min_shape[1]]
                        )

                    st.dataframe(pd.DataFrame(element_product).style.format("{:.2f}"))

                st.markdown(f"**➕ 求和结果**: {step['result']}")
                break

    # 数学推导部分
    st.markdown("---")
    st.markdown("### 📐 输出尺寸计算公式")

    col_formula, col_example = st.columns([1, 1])

    with col_formula:
        st.markdown(
            "$$ H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - K}{S} \\right\\rfloor + 1 $$"
        )
        st.markdown(
            "$$ W_{out} = \\left\\lfloor \\frac{W_{in} + 2P - K}{S} \\right\\rfloor + 1 $$"
        )

        st.markdown("**参数说明：**")
        st.markdown("- $H_{in}, W_{in}$: 输入高宽")
        st.markdown("- $H_{out}, W_{out}$: 输出高宽")
        st.markdown("- $K$: 卷积核大小")
        st.markdown("- $S$: 步长")
        st.markdown("- $P$: 填充大小")
        st.markdown("- $\\left\\lfloor \\cdot \\right\\rfloor$: 向下取整")

    with col_example:
        st.markdown("### 当前参数计算")
        h_out = (input_size + 2 * padding - kernel_size) // stride + 1
        w_out = (input_size + 2 * padding - kernel_size) // stride + 1

        st.markdown(
            f"""
        **输入尺寸**: {input_size} $$ \\times $$ {input_size}
        
        **卷积核**: {kernel_size} $$ \\times $$ {kernel_size}
        
        **步长**: {stride}
        
        **填充**: {padding}
        
        **输出尺寸**: {h_out} $$ \\times $$ {w_out}
        
        **计算过程**:
        $$ H_{{out}} = \left\lfloor \\frac{{H_{{in}} + 2P - K}}{{S}} \\right\\rfloor + 1 \\\
         \\frac{{input_size + 2 \\times padding - kernel_size}}{{stride}} + 1 = {{h_out}}$$
        """
        )


if __name__ == "__main__":
    # 独立运行时的测试
    cnn_tab(True)
