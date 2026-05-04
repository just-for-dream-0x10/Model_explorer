"""
架构计算解剖工作台
Architecture Computational Dissection Workbench

可视化搭建神经网络，深入解剖每一步的数值计算
核心理念：让你看到每一层到底算了什么数值、为什么这样计算、数值如何传播

计算解剖功能：
- 逐层数值计算过程展示
- 参数计算的数学公式推导
- 激活值传播的数值追踪
- 梯度反向传播的数值分析
- 数值稳定性问题的实时检测

v2.2.0 新增：
- 统一稳定性检测
- 参数爆炸预警
- 内存溢出检测
- 瓶颈层识别
"""

import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Optional, Tuple

from utils.memory_analyzer import (
    analyze_conv2d_memory,
    analyze_linear_memory,
    get_tensor_memory,
)
from utils.stability_analyzer import check_activation_stability
from templates.template_loader import TemplateLoader
from utils.template_calculator import TemplateCalculator
from utils.numerical_stability_checker import StabilityChecker


class LayerConfig:
    """层配置（增强版）"""

    def __init__(self, layer_type: str, name: str, params: dict):
        self.layer_type = layer_type
        self.name = name
        self.params = params
        self.output_shape = None
        self.param_count = 0
        self.memory = 0.0
        self.flops = 0

        # 新增：问题检测
        self.has_issues = False
        self.issues = []
        self.warnings = []
        self.recommendations = []

    def to_dict(self) -> dict:
        """转换为字典（用于导出）"""
        return {"layer_type": self.layer_type, "name": self.name, "params": self.params}

    @staticmethod
    def from_dict(data: dict) -> "LayerConfig":
        """从字典创建（用于导入）"""
        return LayerConfig(data["layer_type"], data["name"], data["params"])


def create_layer_from_config(config: LayerConfig, input_shape):
    """根据配置创建PyTorch层"""
    layer_type = config.layer_type
    params = config.params

    try:
        if layer_type == "Conv2d":
            layer = nn.Conv2d(
                params["in_channels"],
                params["out_channels"],
                params["kernel_size"],
                stride=params.get("stride", 1),
                padding=params.get("padding", 0),
            )
            # 计算输出形状
            B, C, H, W = input_shape
            H_out = (
                H + 2 * params.get("padding", 0) - params["kernel_size"]
            ) // params.get("stride", 1) + 1
            W_out = (
                W + 2 * params.get("padding", 0) - params["kernel_size"]
            ) // params.get("stride", 1) + 1
            config.output_shape = (B, params["out_channels"], H_out, W_out)

        elif layer_type == "Linear":
            layer = nn.Linear(params["in_features"], params["out_features"])
            B = input_shape[0]
            config.output_shape = (B, params["out_features"])

        elif layer_type == "ReLU":
            layer = nn.ReLU()
            config.output_shape = input_shape

        elif layer_type == "MaxPool2d":
            layer = nn.MaxPool2d(
                params["kernel_size"],
                stride=params.get("stride", params["kernel_size"]),
            )
            B, C, H, W = input_shape
            H_out = (H - params["kernel_size"]) // params.get(
                "stride", params["kernel_size"]
            ) + 1
            W_out = (W - params["kernel_size"]) // params.get(
                "stride", params["kernel_size"]
            ) + 1
            config.output_shape = (B, C, H_out, W_out)

        elif layer_type == "Flatten":
            layer = nn.Flatten()
            B = input_shape[0]
            flat_size = np.prod(input_shape[1:])
            config.output_shape = (B, int(flat_size))

        elif layer_type == "BatchNorm2d":
            layer = nn.BatchNorm2d(params["num_features"])
            config.output_shape = input_shape

        elif layer_type == "Dropout":
            layer = nn.Dropout(params.get("p", 0.5))
            config.output_shape = input_shape

        else:
            raise ValueError(f"不支持的层类型: {layer_type}")

        # 计算参数量
        config.param_count = sum(p.numel() for p in layer.parameters())

        return layer

    except Exception as e:
        st.error(f"创建层失败: {e}")
        return None


def detect_layer_issues(config: LayerConfig, input_shape, prev_config=None) -> None:
    """
    检测层的潜在问题

    Args:
        config: 当前层配置
        input_shape: 输入形状
        prev_config: 前一层配置（可选）
    """
    config.issues = []
    config.warnings = []
    config.recommendations = []
    config.has_issues = False

    layer_type = config.layer_type
    params = config.params

    # 检测Conv2d问题
    if layer_type == "Conv2d":
        # 检查输入维度
        if len(input_shape) != 4:
            config.issues.append(f"❌ Conv2d需要4D输入，当前: {len(input_shape)}D")
            config.recommendations.append("确保输入是 (Batch, Channels, Height, Width)")
            config.has_issues = True
        else:
            in_channels = input_shape[1]
            expected_in = params["in_channels"]
            if in_channels != expected_in:
                config.issues.append(
                    f"❌ 通道数不匹配: 输入{in_channels}, 期望{expected_in}"
                )
                config.recommendations.append(f"将in_channels改为{in_channels}")
                config.has_issues = True

            # 检查kernel_size是否过大
            H, W = input_shape[2], input_shape[3]
            k = params["kernel_size"]
            p = params.get("padding", 0)
            if k > H + 2 * p or k > W + 2 * p:
                config.warnings.append(f"⚠️ 卷积核({k})可能过大")
                config.recommendations.append("减小kernel_size或增加padding")

        # 检查参数数量
        param_count = (
            params["in_channels"]
            * params["out_channels"]
            * (params["kernel_size"] ** 2)
        )
        if param_count > 1_000_000:
            config.warnings.append(f"⚠️ 参数量较大: {param_count:,}")
            config.recommendations.append("考虑使用分组卷积或减少通道数")

    # 检测Linear问题
    elif layer_type == "Linear":
        if len(input_shape) == 4:
            config.issues.append("❌ Linear层需要2D输入，但输入是4D")
            config.recommendations.append("在Linear前添加Flatten层")
            config.has_issues = True
        elif len(input_shape) == 2:
            in_features = input_shape[1]
            expected_in = params["in_features"]
            if in_features != expected_in:
                config.issues.append(
                    f"❌ 特征数不匹配: 输入{in_features}, 期望{expected_in}"
                )
                config.recommendations.append(f"将in_features改为{in_features}")
                config.has_issues = True

        # 检查参数数量
        param_count = params["in_features"] * params["out_features"]
        if param_count > 10_000_000:
            config.warnings.append(f"⚠️ 参数量非常大: {param_count:,}")
            config.recommendations.append("考虑使用Global Average Pooling减少特征维度")

    # 检测BatchNorm问题
    elif layer_type == "BatchNorm2d":
        if len(input_shape) != 4:
            config.issues.append(f"❌ BatchNorm2d需要4D输入")
            config.has_issues = True
        else:
            num_channels = input_shape[1]
            expected = params["num_features"]
            if num_channels != expected:
                config.issues.append(
                    f"❌ 通道数不匹配: 输入{num_channels}, 期望{expected}"
                )
                config.recommendations.append(f"将num_features改为{num_channels}")
                config.has_issues = True

    # 检测Flatten后接Conv2d的问题
    if layer_type == "Conv2d" and prev_config and prev_config.layer_type == "Flatten":
        config.issues.append("❌ 不能在Flatten后接Conv2d")
        config.recommendations.append("卷积层应该在Flatten之前")
        config.has_issues = True

    # 检测连续的Pooling层
    if (
        layer_type == "MaxPool2d"
        and prev_config
        and prev_config.layer_type == "MaxPool2d"
    ):
        config.warnings.append("⚠️ 连续的Pooling层可能导致信息损失过大")
        config.recommendations.append("考虑在Pooling层之间添加卷积层")


def visualize_network_flow(layers_config):
    """可视化网络流程（增强版，带问题标注）"""
    if not layers_config:
        return None

    fig = go.Figure()

    # 创建流程图节点
    y_pos = 0
    annotations = []

    for i, config in enumerate(layers_config):
        # 节点颜色（根据问题状态）
        if config.has_issues:
            color = "#ffcccb"  # 红色（有错误）
            border_color = "red"
            border_width = 3
        elif config.warnings:
            color = "#fff8dc"  # 黄色（有警告）
            border_color = "orange"
            border_width = 3
        elif config.layer_type in ["Conv2d", "Linear"]:
            color = "lightblue"
            border_color = "black"
            border_width = 2
        elif config.layer_type in ["ReLU", "Sigmoid", "Tanh"]:
            color = "lightgreen"
            border_color = "black"
            border_width = 2
        elif config.layer_type in ["MaxPool2d", "AvgPool2d"]:
            color = "lightyellow"
            border_color = "black"
            border_width = 2
        else:
            color = "lightgray"
            border_color = "black"
            border_width = 2

        # 绘制节点
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[y_pos],
                mode="markers+text",
                marker=dict(
                    size=80,
                    color=color,
                    line=dict(color=border_color, width=border_width),
                ),
                text=f"{config.name}",
                textposition="middle center",
                showlegend=False,
                hovertext=f"{config.layer_type}<br>输出: {config.output_shape}",
                hoverinfo="text",
            )
        )

        # 添加详细信息
        info_text = f"<b>{config.layer_type}</b><br>"
        if config.output_shape:
            info_text += f"输出: {config.output_shape}<br>"
        if config.param_count > 0:
            info_text += f"参数: {config.param_count:,}<br>"
        if config.memory > 0:
            info_text += f"内存: {config.memory:.2f}MB"

        annotations.append(
            dict(
                x=1.2,
                y=y_pos,
                text=info_text,
                showarrow=False,
                xanchor="left",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                borderpad=4,
            )
        )

        # 绘制连接线
        if i < len(layers_config) - 1:
            fig.add_trace(
                go.Scatter(
                    x=[0.5, 0.5],
                    y=[y_pos, y_pos - 1],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        y_pos -= 1

    fig.update_layout(
        title="网络架构流程图",
        xaxis=dict(visible=False, range=[0, 2.5]),
        yaxis=dict(visible=False, range=[y_pos, 1]),
        height=max(400, len(layers_config) * 100),
        annotations=annotations,
        showlegend=False,
        margin=dict(l=20, r=300, t=50, b=20),
    )

    return fig


def simulate_forward_pass(model, input_data):
    """模拟前向传播，收集每层输出"""
    activations = []

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(
                    {
                        "name": name,
                        "output": output.detach().cpu(),
                        "shape": tuple(output.shape),
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                    }
                )

        return hook

    # 注册hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(
            module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten)
        ):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_data)

    # 移除hooks
    for hook in hooks:
        hook.remove()

    return output, activations


def visualize_activation_heatmap(activation_data):
    """可视化激活值热力图"""
    output = activation_data["output"]

    # 如果是4D张量（图像），取第一个样本和第一个通道
    if len(output.shape) == 4:
        heatmap_data = output[0, 0].numpy()
    # 如果是2D张量，直接使用
    elif len(output.shape) == 2:
        heatmap_data = output[0].reshape(-1, 1).numpy()
    else:
        return None

    fig = go.Figure(data=go.Heatmap(z=heatmap_data, colorscale="Viridis"))

    fig.update_layout(title=f"{activation_data['name']} - 激活值热力图", height=400)

    return fig


def export_network_config(layers: List[LayerConfig], input_shape: Tuple) -> str:
    """导出网络配置为JSON"""
    config_dict = {
        "input_shape": input_shape,
        "layers": [layer.to_dict() for layer in layers],
    }
    return json.dumps(config_dict, indent=2)


def import_network_config(json_str: str) -> Tuple[List[LayerConfig], Tuple]:
    """从JSON导入网络配置"""
    config_dict = json.loads(json_str)
    layers = [LayerConfig.from_dict(layer_data) for layer_data in config_dict["layers"]]
    input_shape = tuple(config_dict["input_shape"])
    return layers, input_shape


def architecture_designer_tab(chinese_supported=True):
    """架构设计工作台主函数（增强版）"""

    st.header("🎨 架构计算解剖工作台")
    st.markdown(
        """
    > **核心功能**：深入解剖神经网络每一步的数值计算过程
    
    **计算解剖维度**：
    - 🔢 **数值计算公式**：每层的具体数学计算过程
    - 📊 **数值传播追踪**：激活值如何逐层变化
    - 🧮 **参数计算推导**：为什么是这个参数量？
    - 🌊 **梯度数值分析**：梯度如何反向传播
    - ⚠️ **数值稳定性**：什么时候会出现数值问题？
    """
    )

    st.markdown("---")

    # 初始化session state
    if "layers" not in st.session_state:
        st.session_state.layers = []
    if "input_shape" not in st.session_state:
        st.session_state.input_shape = (1, 3, 224, 224)

    # 左右分栏
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🔧 层配置")

        # 输入形状设置
        with st.expander("⚙️ 输入配置", expanded=True):
            # 自动检测输入类型
            current_shape = st.session_state.input_shape
            if len(current_shape) == 4:
                default_input_type = "图像"
            else:
                default_input_type = "向量"

            input_type_index = 0 if default_input_type == "图像" else 1
            input_type = st.selectbox(
                "输入类型",
                ["图像", "向量"],
                index=input_type_index,
                key="input_type_selector",
            )

            if input_type == "图像":
                col1, col2 = st.columns(2)
                with col1:
                    # 从当前形状读取默认值
                    default_channels = (
                        current_shape[1] if len(current_shape) == 4 else 3
                    )
                    channels = st.number_input(
                        "通道数", 1, 4, default_channels, key="input_channels"
                    )
                with col2:
                    default_size = current_shape[2] if len(current_shape) == 4 else 224
                    size_options = [28, 32, 64, 224]
                    default_index = (
                        size_options.index(default_size)
                        if default_size in size_options
                        else 3
                    )
                    img_size = st.selectbox(
                        "图像尺寸", size_options, index=default_index, key="input_size"
                    )
                st.session_state.input_shape = (1, channels, img_size, img_size)
            else:
                # 从当前形状读取默认值
                default_vector_size = (
                    current_shape[1] if len(current_shape) == 2 else 784
                )
                vector_size = st.number_input(
                    "向量维度", 1, 100000, default_vector_size, key="vector_size"
                )
                st.session_state.input_shape = (1, vector_size)

            st.info(f"当前输入形状: `{st.session_state.input_shape}`")

        # 添加层
        with st.expander("➕ 添加层", expanded=True):
            layer_type = st.selectbox(
                "选择层类型",
                [
                    "Conv2d",
                    "Linear",
                    "ReLU",
                    "MaxPool2d",
                    "Flatten",
                    "BatchNorm2d",
                    "Dropout",
                ],
            )

            layer_name = st.text_input(
                "层名称", f"{layer_type}_{len(st.session_state.layers)+1}"
            )

            params = {}

            if layer_type == "Conv2d":
                col1, col2 = st.columns(2)
                with col1:
                    # 智能默认值：自动匹配前一层的输出
                    if len(st.session_state.layers) == 0:
                        # 第一层：匹配输入通道数
                        default_in_channels = st.session_state.input_shape[1]
                    else:
                        # 后续层：匹配前一层输出
                        prev_layer = st.session_state.layers[-1]
                        if prev_layer.layer_type == "Conv2d":
                            default_in_channels = prev_layer.params.get(
                                "out_channels", 64
                            )
                        elif prev_layer.layer_type == "BatchNorm2d":
                            default_in_channels = prev_layer.params.get(
                                "num_features", 64
                            )
                        elif (
                            prev_layer.output_shape
                            and len(prev_layer.output_shape) == 4
                        ):
                            default_in_channels = prev_layer.output_shape[1]
                        else:
                            default_in_channels = 64
                    params["in_channels"] = st.number_input(
                        "输入通道", 1, 512, default_in_channels
                    )
                    params["out_channels"] = st.number_input("输出通道", 1, 512, 64)
                with col2:
                    params["kernel_size"] = st.number_input("卷积核大小", 1, 11, 3)
                    params["stride"] = st.number_input("步长", 1, 4, 1)
                    params["padding"] = st.number_input("填充", 0, 10, 1)

            elif layer_type == "Linear":
                # 智能默认值：自动计算 Flatten 后的特征数
                if len(st.session_state.layers) > 0:
                    prev_layer = st.session_state.layers[-1]
                    if prev_layer.layer_type == "Flatten" and prev_layer.output_shape:
                        # Flatten 后的输出是 (batch, features)
                        default_in_features = prev_layer.output_shape[1]
                    elif prev_layer.layer_type == "Linear":
                        default_in_features = prev_layer.params.get("out_features", 128)
                    elif prev_layer.output_shape and len(prev_layer.output_shape) == 2:
                        default_in_features = prev_layer.output_shape[1]
                    else:
                        default_in_features = 784
                else:
                    default_in_features = 784

                # 显示提示信息
                if (
                    len(st.session_state.layers) > 0
                    and st.session_state.layers[-1].layer_type == "Flatten"
                ):
                    st.info(f"💡 Flatten后特征数: {default_in_features:,}")

                params["in_features"] = st.number_input(
                    "输入特征", 1, 10000000, default_in_features
                )
                params["out_features"] = st.number_input("输出特征", 1, 10000, 128)

            elif layer_type == "MaxPool2d":
                params["kernel_size"] = st.number_input("池化核大小", 2, 8, 2)
                params["stride"] = st.number_input("步长", 1, 8, 2)

            elif layer_type == "BatchNorm2d":
                # 智能默认值：匹配前一层的通道数
                if len(st.session_state.layers) > 0:
                    prev_layer = st.session_state.layers[-1]
                    if prev_layer.layer_type == "Conv2d":
                        default_num_features = prev_layer.params.get("out_channels", 64)
                    elif prev_layer.output_shape and len(prev_layer.output_shape) == 4:
                        default_num_features = prev_layer.output_shape[1]
                    else:
                        default_num_features = 64
                else:
                    default_num_features = 64
                params["num_features"] = st.number_input(
                    "特征数", 1, 512, default_num_features
                )

            elif layer_type == "Dropout":
                params["p"] = st.slider("丢弃率", 0.0, 0.9, 0.5)

            if st.button("➕ 添加到网络"):
                config = LayerConfig(layer_type, layer_name, params)
                st.session_state.layers.append(config)
                st.success(f"✅ 已添加 {layer_name}")
                st.rerun()

        # 层管理（增强版：带上下移动）
        if st.session_state.layers:
            with st.expander("📋 层管理", expanded=True):
                for i, layer in enumerate(st.session_state.layers):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        # 显示层名和状态图标
                        status_icon = (
                            "❌"
                            if layer.has_issues
                            else ("⚠️" if layer.warnings else "✅")
                        )
                        st.write(
                            f"{i+1}. {status_icon} {layer.name} ({layer.layer_type})"
                        )
                    with col2:
                        if i > 0 and st.button("⬆️", key=f"up_{i}", help="上移"):
                            (
                                st.session_state.layers[i],
                                st.session_state.layers[i - 1],
                            ) = (
                                st.session_state.layers[i - 1],
                                st.session_state.layers[i],
                            )
                            st.rerun()
                    with col3:
                        if i < len(st.session_state.layers) - 1 and st.button(
                            "⬇️", key=f"down_{i}", help="下移"
                        ):
                            (
                                st.session_state.layers[i],
                                st.session_state.layers[i + 1],
                            ) = (
                                st.session_state.layers[i + 1],
                                st.session_state.layers[i],
                            )
                            st.rerun()
                    with col4:
                        if st.button("🗑️", key=f"del_{i}", help="删除"):
                            st.session_state.layers.pop(i)
                            st.rerun()

        # 导入/导出配置
        with st.expander("💾 导入/导出", expanded=False):
            st.markdown("**导出配置**")
            if st.session_state.layers:
                config_json = export_network_config(
                    st.session_state.layers, st.session_state.input_shape
                )
                st.download_button(
                    label="📥 下载配置文件",
                    data=config_json,
                    file_name="network_config.json",
                    mime="application/json",
                )
                st.code(config_json, language="json")
            else:
                st.info("暂无网络配置")

            st.markdown("---")
            st.markdown("**导入配置**")
            uploaded_config = st.file_uploader("上传配置文件 (JSON)", type=["json"])
            if uploaded_config:
                try:
                    config_json = uploaded_config.read().decode("utf-8")
                    layers, input_shape = import_network_config(config_json)
                    if st.button("✅ 应用配置"):
                        st.session_state.layers = layers
                        st.session_state.input_shape = input_shape
                        st.success("配置已导入！")
                        st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {e}")

    with col_right:
        st.subheader("📊 网络分析")

        if not st.session_state.layers:
            st.info("👈 从左侧添加层开始构建网络")
        else:
            # 🔄 强制重新计算所有层的输出形状（确保数据同步）
            current_shape = st.session_state.input_shape
            valid_network = True
            has_any_issues = False
            has_any_warnings = False

            # 第一遍：重新计算所有输出形状
            for idx, config in enumerate(st.session_state.layers):
                try:
                    layer = create_layer_from_config(config, current_shape)
                    if layer is None:
                        valid_network = False
                        break
                    current_shape = config.output_shape
                except Exception as e:
                    valid_network = False
                    break

            # 第二遍：检测问题
            current_shape = st.session_state.input_shape
            for idx, config in enumerate(st.session_state.layers):
                prev_config = st.session_state.layers[idx - 1] if idx > 0 else None

                try:
                    # 检测问题
                    detect_layer_issues(config, current_shape, prev_config)

                    if config.has_issues:
                        has_any_issues = True
                    if config.warnings:
                        has_any_warnings = True

                    # 更新当前形状
                    if config.output_shape:
                        current_shape = config.output_shape
                        # 计算内存
                        config.memory = get_tensor_memory(config.output_shape)
                    else:
                        valid_network = False
                        break

                except Exception as e:
                    st.error(f"层 {config.name} 配置错误: {e}")
                    valid_network = False
                    break

            # 显示问题摘要
            if has_any_issues or has_any_warnings:
                st.markdown("#### ⚠️ 问题检测")

                issue_count = sum(1 for c in st.session_state.layers if c.has_issues)
                warning_count = sum(
                    1
                    for c in st.session_state.layers
                    if c.warnings and not c.has_issues
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if issue_count > 0:
                        st.error(f"❌ 发现 {issue_count} 个错误")
                with col2:
                    if warning_count > 0:
                        st.warning(f"⚠️ 发现 {warning_count} 个警告")
                with col3:
                    if issue_count > 0 and st.button(
                        "🔧 一键修复", type="primary", help="自动修正所有形状不匹配问题"
                    ):
                        # 自动修复所有层 - 改进版（支持自动插入Flatten）
                        temp_shape = st.session_state.input_shape
                        fixed_count = 0
                        new_layers = []

                        for idx, config in enumerate(st.session_state.layers):
                            # 检查是否需要插入Flatten层
                            if config.layer_type == "Linear" and len(temp_shape) == 4:
                                # 需要在Linear前插入Flatten
                                flatten_config = LayerConfig(
                                    "Flatten", f"flatten_auto_{idx}", {}
                                )
                                flatten_layer = create_layer_from_config(
                                    flatten_config, temp_shape
                                )
                                new_layers.append(flatten_config)
                                temp_shape = flatten_config.output_shape
                                fixed_count += 1

                            # 根据当前形状修复参数
                            if config.layer_type == "Conv2d" and len(temp_shape) == 4:
                                if config.params["in_channels"] != temp_shape[1]:
                                    config.params["in_channels"] = temp_shape[1]
                                    fixed_count += 1

                            elif (
                                config.layer_type == "BatchNorm2d"
                                and len(temp_shape) == 4
                            ):
                                if config.params["num_features"] != temp_shape[1]:
                                    config.params["num_features"] = temp_shape[1]
                                    fixed_count += 1

                            elif config.layer_type == "Linear" and len(temp_shape) == 2:
                                if config.params["in_features"] != temp_shape[1]:
                                    config.params["in_features"] = temp_shape[1]
                                    fixed_count += 1

                            # 添加当前层
                            new_layers.append(config)

                            # 重新计算输出形状（使用修复后的参数）
                            try:
                                layer = create_layer_from_config(config, temp_shape)
                                if config.output_shape:
                                    temp_shape = config.output_shape
                            except Exception as e:
                                # 如果仍然失败，记录错误但继续
                                pass

                        # 更新层列表
                        st.session_state.layers = new_layers

                        if fixed_count > 0:
                            st.success(f"✅ 自动修复完成！已修正 {fixed_count} 个问题")
                        else:
                            st.info("ℹ️ 没有找到需要修复的参数")
                        st.rerun()

                # 显示详细问题
                for i, config in enumerate(st.session_state.layers):
                    if config.issues or config.warnings:
                        with st.expander(
                            f"{'❌' if config.has_issues else '⚠️'} {config.name} - 问题详情",
                            expanded=config.has_issues,
                        ):
                            if config.issues:
                                st.markdown("**错误：**")
                                for issue in config.issues:
                                    st.markdown(f"- {issue}")

                            if config.warnings:
                                st.markdown("**警告：**")
                                for warning in config.warnings:
                                    st.markdown(f"- {warning}")

                            if config.recommendations:
                                st.markdown("**建议：**")
                                for rec in config.recommendations:
                                    st.markdown(f"- 💡 {rec}")

                st.markdown("---")

            if valid_network:
                # 显示网络流程图
                fig = visualize_network_flow(st.session_state.layers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # 显示总结
                st.markdown("#### 📋 网络总结")

                total_params = sum(
                    config.param_count for config in st.session_state.layers
                )
                total_memory = sum(config.memory for config in st.session_state.layers)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总层数", len(st.session_state.layers))
                with col2:
                    st.metric("总参数量", f"{total_params:,}")
                with col3:
                    st.metric("激活内存", f"{total_memory:.2f} MB")
                with col4:
                    st.metric("输出形状", str(current_shape))

                # ==================== 网络稳定性诊断 ====================
                st.markdown("---")
                st.markdown("#### 🔬 网络稳定性诊断")

                stability_issues = []

                # 1. 检查总参数量
                if total_params > 1e9:
                    stability_issues.append(
                        {
                            "status": "error",
                            "type": "参数量过大",
                            "value": f"{total_params/1e9:.2f}B",
                            "threshold": "> 1B",
                            "icon": "🔴",
                            "severity": "critical",
                            "details": {
                                "总参数": f"{total_params:,}",
                                "层数": len(st.session_state.layers),
                            },
                            "solution": [
                                "使用深度可分离卷积减少参数",
                                "减少全连接层神经元数量",
                                "使用MobileNet/EfficientNet架构",
                                "添加更多Pooling层",
                            ],
                            "explanation": "参数量过大会导致显存不足、训练慢、容易过拟合",
                        }
                    )
                elif total_params > 1e8:
                    stability_issues.append(
                        {
                            "status": "warning",
                            "type": "参数量较大",
                            "value": f"{total_params/1e6:.1f}M",
                            "threshold": "> 100M",
                            "icon": "🟡",
                            "severity": "medium",
                            "details": {
                                "总参数": f"{total_params:,}",
                                "估算显存": f"{total_params * 4 / 1024 / 1024:.1f} MB",
                            },
                            "solution": [
                                "监控显存使用",
                                "考虑使用混合精度训练",
                                "适当减少batch size",
                            ],
                            "explanation": "参数量较大，注意显存管理",
                        }
                    )
                else:
                    stability_issues.append(
                        {
                            "status": "success",
                            "type": "参数量",
                            "value": (
                                f"{total_params/1e6:.2f}M"
                                if total_params > 1e6
                                else f"{total_params:,}"
                            ),
                            "icon": "🟢",
                            "severity": "none",
                        }
                    )

                # 2. 检查激活内存
                if total_memory > 1000:
                    stability_issues.append(
                        {
                            "status": "error",
                            "type": "激活内存过大",
                            "value": f"{total_memory:.1f} MB",
                            "threshold": "> 1000 MB",
                            "icon": "🔴",
                            "severity": "high",
                            "details": {
                                "激活内存": f"{total_memory:.1f} MB",
                                "估算总显存": f"{total_memory * 3:.1f} MB (含梯度)",
                            },
                            "solution": [
                                "减小batch size",
                                "增加Pooling层",
                                "使用梯度检查点(gradient checkpointing)",
                                "减少输入图像尺寸",
                            ],
                            "explanation": "激活内存过大会导致OOM（显存溢出）",
                        }
                    )
                elif total_memory > 500:
                    stability_issues.append(
                        {
                            "status": "warning",
                            "type": "激活内存较大",
                            "value": f"{total_memory:.1f} MB",
                            "threshold": "> 500 MB",
                            "icon": "🟡",
                            "severity": "medium",
                            "solution": ["监控显存使用", "考虑减小batch size"],
                            "explanation": "激活内存较大，注意batch size设置",
                        }
                    )

                # 3. 识别瓶颈层
                if st.session_state.layers:
                    max_params_layer = max(
                        st.session_state.layers, key=lambda x: x.param_count
                    )
                    max_memory_layer = max(
                        st.session_state.layers, key=lambda x: x.memory
                    )

                    params_ratio = max_params_layer.param_count / (total_params + 1)
                    memory_ratio = max_memory_layer.memory / (total_memory + 1)

                    if params_ratio > 0.5:
                        stability_issues.append(
                            {
                                "status": "warning",
                                "type": "参数瓶颈层",
                                "value": f"{max_params_layer.name} ({params_ratio*100:.1f}%)",
                                "threshold": "> 50%",
                                "icon": "🟡",
                                "severity": "medium",
                                "details": {
                                    "瓶颈层": max_params_layer.name,
                                    "参数量": f"{max_params_layer.param_count:,}",
                                    "占比": f"{params_ratio*100:.1f}%",
                                },
                                "solution": [
                                    "减少该层的神经元数量",
                                    "使用参数分解技术",
                                    "考虑使用瓶颈结构",
                                ],
                                "explanation": f"{max_params_layer.name}层占用了超过一半的参数",
                            }
                        )

                    if memory_ratio > 0.5:
                        stability_issues.append(
                            {
                                "status": "warning",
                                "type": "内存瓶颈层",
                                "value": f"{max_memory_layer.name} ({memory_ratio*100:.1f}%)",
                                "threshold": "> 50%",
                                "icon": "🟡",
                                "severity": "medium",
                                "details": {
                                    "瓶颈层": max_memory_layer.name,
                                    "内存": f"{max_memory_layer.memory:.2f} MB",
                                    "占比": f"{memory_ratio*100:.1f}%",
                                },
                                "solution": [
                                    "在该层之前添加Pooling",
                                    "减少该层的通道数",
                                    "使用深度可分离卷积",
                                ],
                                "explanation": f"{max_memory_layer.name}层占用了超过一半的激活内存",
                            }
                        )

                # 4. 检查网络深度
                conv_count = sum(
                    1 for c in st.session_state.layers if "Conv" in c.layer_type
                )
                fc_count = sum(
                    1 for c in st.session_state.layers if c.layer_type == "Linear"
                )

                if conv_count > 50:
                    stability_issues.append(
                        {
                            "status": "warning",
                            "type": "网络过深",
                            "value": f"{conv_count}层卷积",
                            "threshold": "> 50层",
                            "icon": "🟡",
                            "severity": "medium",
                            "solution": [
                                "使用残差连接（ResNet）",
                                "使用BatchNorm稳定训练",
                                "考虑使用DenseNet或其他skip connection",
                            ],
                            "explanation": "深层网络容易出现梯度消失，需要特殊设计",
                        }
                    )

                # 显示诊断结果
                if stability_issues:
                    StabilityChecker.display_issues(
                        stability_issues, title="🔬 网络架构稳定性诊断"
                    )
                else:
                    st.success("✅ 网络架构检查通过，未发现问题")

                # 显示各层详细信息表格
                if st.checkbox("📊 显示详细层信息表", value=False):
                    import pandas as pd

                    table_data = []
                    for i, config in enumerate(st.session_state.layers):
                        status = (
                            "❌ 错误"
                            if config.has_issues
                            else ("⚠️ 警告" if config.warnings else "✅ 正常")
                        )
                        table_data.append(
                            {
                                "序号": i + 1,
                                "层名": config.name,
                                "类型": config.layer_type,
                                "输出形状": (
                                    str(config.output_shape)
                                    if config.output_shape
                                    else "N/A"
                                ),
                                "参数量": f"{config.param_count:,}",
                                "内存(MB)": f"{config.memory:.2f}",
                                "状态": status,
                            }
                        )

                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # 生成代码
                st.markdown("---")
                st.markdown("#### 💻 生成PyTorch代码")

                code = "import torch.nn as nn\n\nclass CustomModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n"

                for config in st.session_state.layers:
                    if config.layer_type == "Conv2d":
                        code += f"        self.{config.name} = nn.Conv2d({config.params['in_channels']}, {config.params['out_channels']}, {config.params['kernel_size']}, stride={config.params.get('stride', 1)}, padding={config.params.get('padding', 0)})\n"
                    elif config.layer_type == "Linear":
                        code += f"        self.{config.name} = nn.Linear({config.params['in_features']}, {config.params['out_features']})\n"
                    elif config.layer_type == "ReLU":
                        code += f"        self.{config.name} = nn.ReLU()\n"
                    elif config.layer_type == "MaxPool2d":
                        code += f"        self.{config.name} = nn.MaxPool2d({config.params['kernel_size']}, stride={config.params.get('stride', config.params['kernel_size'])})\n"
                    elif config.layer_type == "Flatten":
                        code += f"        self.{config.name} = nn.Flatten()\n"
                    elif config.layer_type == "BatchNorm2d":
                        code += f"        self.{config.name} = nn.BatchNorm2d({config.params['num_features']})\n"
                    elif config.layer_type == "Dropout":
                        code += f"        self.{config.name} = nn.Dropout({config.params.get('p', 0.5)})\n"

                code += "\n    def forward(self, x):\n"
                for config in st.session_state.layers:
                    code += f"        x = self.{config.name}(x)\n"
                code += "        return x"

                st.code(code, language="python")

                # 前向传播模拟
                st.markdown("---")
                st.markdown("#### 🚀 前向传播模拟")

                st.info("上传图片或使用随机数据测试网络")

                col1, col2 = st.columns(2)

                with col1:
                    use_random = st.checkbox("使用随机数据", value=True)

                with col2:
                    if not use_random:
                        uploaded_file = st.file_uploader(
                            "上传图片", type=["png", "jpg", "jpeg"]
                        )

                # 运行前检查是否有错误
                has_errors_before_run = any(
                    c.has_issues for c in st.session_state.layers
                )

                if has_errors_before_run:
                    st.error("⚠️ 网络中存在错误，无法运行前向传播！请先修复错误。")
                    if st.button(
                        "🔧 自动修复并运行", type="primary", key="auto_fix_run"
                    ):
                        # 自动修复 - 改进版（支持自动插入Flatten）
                        temp_shape = st.session_state.input_shape
                        fixed_count = 0
                        new_layers = []

                        for idx, config in enumerate(st.session_state.layers):
                            # 检查是否需要插入Flatten层
                            if config.layer_type == "Linear" and len(temp_shape) == 4:
                                # 需要在Linear前插入Flatten
                                flatten_config = LayerConfig(
                                    "Flatten", f"flatten_auto_{idx}", {}
                                )
                                flatten_layer = create_layer_from_config(
                                    flatten_config, temp_shape
                                )
                                new_layers.append(flatten_config)
                                temp_shape = flatten_config.output_shape
                                fixed_count += 1

                            # 根据当前形状修复参数
                            if config.layer_type == "Conv2d" and len(temp_shape) == 4:
                                if config.params["in_channels"] != temp_shape[1]:
                                    config.params["in_channels"] = temp_shape[1]
                                    fixed_count += 1

                            elif (
                                config.layer_type == "BatchNorm2d"
                                and len(temp_shape) == 4
                            ):
                                if config.params["num_features"] != temp_shape[1]:
                                    config.params["num_features"] = temp_shape[1]
                                    fixed_count += 1

                            elif config.layer_type == "Linear" and len(temp_shape) == 2:
                                if config.params["in_features"] != temp_shape[1]:
                                    config.params["in_features"] = temp_shape[1]
                                    fixed_count += 1

                            # 添加当前层
                            new_layers.append(config)

                            # 重新计算输出形状（使用修复后的参数）
                            try:
                                layer = create_layer_from_config(config, temp_shape)
                                if config.output_shape:
                                    temp_shape = config.output_shape
                            except Exception:
                                pass

                        # 更新层列表
                        st.session_state.layers = new_layers

                        st.success(
                            f"✅ 自动修复完成！已修正 {fixed_count} 个问题，正在运行..."
                        )
                        st.rerun()

                elif st.button("▶️ 运行前向传播", type="primary"):
                    with st.spinner("计算中..."):
                        try:
                            # 构建模型
                            layers_list = []
                            current_shape = st.session_state.input_shape

                            for config in st.session_state.layers:
                                layer = create_layer_from_config(config, current_shape)
                                if layer:
                                    layers_list.append(layer)
                                    current_shape = config.output_shape

                            model = nn.Sequential(*layers_list)

                            # 准备输入
                            if use_random:
                                input_data = torch.randn(st.session_state.input_shape)
                            else:
                                # 处理上传的图片
                                if uploaded_file is not None:
                                    image = Image.open(uploaded_file).convert("RGB")
                                    img_array = np.array(image, dtype=np.float32)
                                    if img_array.max() > 1.0:
                                        img_array = img_array / 255.0
                                    # HWC -> CHW
                                    img_tensor = torch.from_numpy(
                                        img_array.transpose(2, 0, 1)
                                    )
                                    # 调整到期望的输入形状 (batch_size, C, H, W)
                                    expected_shape = st.session_state.input_shape
                                    if len(expected_shape) == 3:
                                        input_data = img_tensor.unsqueeze(0)
                                        # 如果通道数不匹配，取灰度
                                        if input_data.shape[1] != expected_shape[0]:
                                            gray = Image.open(uploaded_file).convert("L")
                                            gray_array = np.array(gray, dtype=np.float32)
                                            if gray_array.max() > 1.0:
                                                gray_array = gray_array / 255.0
                                            input_data = torch.from_numpy(
                                                gray_array
                                            ).unsqueeze(0).unsqueeze(0)
                                            # 复制到目标通道数
                                            input_data = input_data.expand(
                                                -1, expected_shape[0], -1, -1
                                            )
                                        # 调整空间尺寸
                                        if (
                                            input_data.shape[2] != expected_shape[1]
                                            or input_data.shape[3] != expected_shape[2]
                                        ):
                                            input_data = torch.nn.functional.interpolate(
                                                input_data,
                                                size=(expected_shape[1], expected_shape[2]),
                                                mode="bilinear",
                                                align_corners=False,
                                            )
                                    else:
                                        input_data = torch.randn(
                                            st.session_state.input_shape
                                        )
                                        st.warning(
                                            "不支持的输入形状，已回退为随机数据"
                                        )
                                else:
                                    st.warning("请先上传图片，或勾选'使用随机数据'")
                                    st.stop()

                            # 前向传播
                            output, activations = simulate_forward_pass(
                                model, input_data
                            )

                            st.success("✅ 前向传播完成！")

                            # 显示输出
                            st.markdown("#### 📊 输出结果")
                            st.write(f"最终输出形状: `{tuple(output.shape)}`")

                            if output.numel() <= 100:
                                st.write("输出值:")
                                st.write(output.squeeze().numpy())
                            else:
                                st.write(
                                    f"输出统计: mean={output.mean():.4f}, std={output.std():.4f}, min={output.min():.4f}, max={output.max():.4f}"
                                )

                            # 显示每层激活值
                            if activations:
                                st.markdown("---")
                                st.markdown("#### 🔍 逐层激活值分析")

                                for act in activations:
                                    with st.expander(
                                        f"📍 {act['name']} - 形状: {act['shape']}",
                                        expanded=False,
                                    ):
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.markdown("**统计信息**")
                                            st.write(f"均值: {act['mean']:.4f}")
                                            st.write(f"标准差: {act['std']:.4f}")
                                            st.write(f"最小值: {act['min']:.4f}")
                                            st.write(f"最大值: {act['max']:.4f}")

                                        with col2:
                                            st.markdown("**形状信息**")
                                            st.write(f"输出形状: {act['shape']}")
                                            st.write(
                                                f"元素数量: {np.prod(act['shape']):,}"
                                            )

                                        # 可视化
                                        fig = visualize_activation_heatmap(act)
                                        if fig:
                                            st.plotly_chart(
                                                fig, use_container_width=True
                                            )

                        except Exception as e:
                            st.error(f"前向传播失败: {e}")
                            import traceback

                            st.code(traceback.format_exc())

    # 底部提示
    st.markdown("---")

    # 显示快捷模板（使用新的模板系统）
    with st.expander("🚀 神经网络模板库", expanded=False):
        st.markdown("### 📚 预设网络架构模板")
        st.markdown("从12+种经典架构中选择，一键加载完整网络配置")

        # 初始化模板加载器
        try:
            loader = TemplateLoader()
            templates = loader.get_all_templates()

            if not templates:
                st.warning("⚠️ 未找到模板文件，请确保 templates/configs/ 目录存在")
            else:
                # 按分类显示模板
                categories = loader.get_categories()

                # 添加筛选选项
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                with col_filter1:
                    selected_category = st.selectbox(
                        "📂 按分类筛选",
                        ["全部"] + categories,
                        key="template_category_filter",
                    )
                with col_filter2:
                    selected_difficulty = st.selectbox(
                        "📊 按难度筛选",
                        ["全部", "beginner", "intermediate", "advanced"],
                        format_func=lambda x: {
                            "全部": "全部",
                            "beginner": "入门",
                            "intermediate": "中级",
                            "advanced": "高级",
                        }.get(x, x),
                        key="template_difficulty_filter",
                    )
                with col_filter3:
                    search_query = st.text_input(
                        "🔍 搜索模板",
                        placeholder="输入关键词...",
                        key="template_search",
                    )

                # 应用筛选
                filtered_templates = templates
                if selected_category != "全部":
                    filtered_templates = [
                        t for t in filtered_templates if t.category == selected_category
                    ]
                if selected_difficulty != "全部":
                    filtered_templates = [
                        t
                        for t in filtered_templates
                        if t.difficulty == selected_difficulty
                    ]
                if search_query:
                    filtered_templates = loader.search_templates(search_query)

                if not filtered_templates:
                    st.info("没有找到匹配的模板")
                else:
                    st.markdown(f"**找到 {len(filtered_templates)} 个模板**")

                    # 按分类组织显示
                    for category in categories:
                        cat_templates = [
                            t for t in filtered_templates if t.category == category
                        ]
                        if not cat_templates:
                            continue

                        st.markdown(f"#### 📁 {category}")

                        # 每行显示3个模板
                        for i in range(0, len(cat_templates), 3):
                            cols = st.columns(3)
                            for j, col in enumerate(cols):
                                if i + j < len(cat_templates):
                                    template = cat_templates[i + j]
                                    with col:
                                        # 难度标签
                                        difficulty_colors = {
                                            "beginner": "🟢",
                                            "intermediate": "🟡",
                                            "advanced": "🔴",
                                        }
                                        difficulty_emoji = difficulty_colors.get(
                                            template.difficulty, "⚪"
                                        )

                                        # 创建按钮
                                        button_label = f"{template.icon} {template.name}\n{difficulty_emoji}"
                                        if st.button(
                                            button_label,
                                            key=f"template_{template.id}",
                                            use_container_width=True,
                                            help=f"{template.description}\n层数: {len(template.layers)}\n输入: {template.input_shape}",
                                        ):
                                            # 加载模板
                                            st.session_state.input_shape = tuple(
                                                template.input_shape
                                            )
                                            st.session_state.layers = (
                                                template.to_layer_configs()
                                            )
                                            st.success(f"✅ 已加载 {template.name}")
                                            st.info(f"📋 {template.description}")
                                            st.rerun()

                                        # 显示简要信息
                                        st.caption(
                                            f"{len(template.layers)} 层 | {template.input_shape}"
                                        )

                        st.markdown("---")

        except Exception as e:
            st.error(f"加载模板失败: {e}")
            st.info("💡 使用默认模板作为备选...")

            # 备选方案：显示旧的硬编码模板
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📱 简单CNN (MNIST)", use_container_width=True):
                    st.session_state.input_shape = (1, 1, 28, 28)
                    st.session_state.layers = [
                        LayerConfig(
                            "Conv2d",
                            "conv1",
                            {
                                "in_channels": 1,
                                "out_channels": 32,
                                "kernel_size": 3,
                                "stride": 1,
                                "padding": 1,
                            },
                        ),
                        LayerConfig("ReLU", "relu1", {}),
                        LayerConfig(
                            "MaxPool2d", "pool1", {"kernel_size": 2, "stride": 2}
                        ),
                        LayerConfig(
                            "Conv2d",
                            "conv2",
                            {
                                "in_channels": 32,
                                "out_channels": 64,
                                "kernel_size": 3,
                                "stride": 1,
                                "padding": 1,
                            },
                        ),
                        LayerConfig("ReLU", "relu2", {}),
                        LayerConfig(
                            "MaxPool2d", "pool2", {"kernel_size": 2, "stride": 2}
                        ),
                        LayerConfig("Flatten", "flatten", {}),
                        LayerConfig(
                            "Linear", "fc1", {"in_features": 3136, "out_features": 128}
                        ),
                        LayerConfig("ReLU", "relu3", {}),
                        LayerConfig(
                            "Linear", "fc2", {"in_features": 128, "out_features": 10}
                        ),
                    ]
                    st.success("✅ 已加载 MNIST CNN 模板")
                    st.rerun()

            with col2:
                if st.button("🖼️ 中等CNN (CIFAR)", use_container_width=True):
                    st.session_state.input_shape = (1, 3, 32, 32)
                    st.session_state.layers = [
                        LayerConfig(
                            "Conv2d",
                            "conv1",
                            {
                                "in_channels": 3,
                                "out_channels": 64,
                                "kernel_size": 3,
                                "stride": 1,
                                "padding": 1,
                            },
                        ),
                        LayerConfig("BatchNorm2d", "bn1", {"num_features": 64}),
                        LayerConfig("ReLU", "relu1", {}),
                        LayerConfig(
                            "Conv2d",
                            "conv2",
                            {
                                "in_channels": 64,
                                "out_channels": 128,
                                "kernel_size": 3,
                                "stride": 1,
                                "padding": 1,
                            },
                        ),
                        LayerConfig("BatchNorm2d", "bn2", {"num_features": 128}),
                        LayerConfig("ReLU", "relu2", {}),
                        LayerConfig(
                            "MaxPool2d", "pool1", {"kernel_size": 2, "stride": 2}
                        ),
                        LayerConfig("Flatten", "flatten", {}),
                        LayerConfig(
                            "Linear", "fc1", {"in_features": 32768, "out_features": 256}
                        ),
                        LayerConfig("ReLU", "relu3", {}),
                        LayerConfig("Dropout", "dropout", {"p": 0.5}),
                        LayerConfig(
                            "Linear", "fc2", {"in_features": 256, "out_features": 10}
                        ),
                    ]
                    st.success("✅ 已加载 CIFAR CNN 模板")
                    st.rerun()

            with col3:
                if st.button("🧠 简单MLP", use_container_width=True):
                    st.session_state.input_shape = (1, 784)
                    st.session_state.layers = [
                        LayerConfig(
                            "Linear", "fc1", {"in_features": 784, "out_features": 512}
                        ),
                        LayerConfig("ReLU", "relu1", {}),
                        LayerConfig("Dropout", "dropout1", {"p": 0.2}),
                        LayerConfig(
                            "Linear", "fc2", {"in_features": 512, "out_features": 256}
                        ),
                        LayerConfig("ReLU", "relu2", {}),
                        LayerConfig("Dropout", "dropout2", {"p": 0.2}),
                        LayerConfig(
                            "Linear", "fc3", {"in_features": 256, "out_features": 10}
                        ),
                    ]
                    st.success("✅ 已加载 MLP 模板")
                    st.rerun()

    st.markdown(
        """
    ### 💡 使用提示
    
    1. **从输入开始** - 先配置输入形状（图像或向量）
    2. **逐层添加** - 从左侧添加层，注意形状匹配
    3. **实时反馈** - 每添加一层，立即看到输出形状和参数量
    4. **自动检测** - 系统会自动检测形状不匹配、参数过大等问题
    5. **层重排序** - 使用⬆️⬇️按钮调整层的顺序
    6. **模拟运行** - 点击"运行前向传播"查看逐层计算结果
    7. **保存配置** - 导出配置文件以便后续使用
    
    ### ⚠️ 常见问题
    
    - **形状不匹配**：检查前一层的输出是否符合当前层的输入要求（系统会自动提示）
    - **参数过多**：在Flatten后接Linear时注意特征维度，考虑使用Pooling减少尺寸
    - **通道数错误**：Conv2d和BatchNorm2d的通道数要匹配
    - **红色节点**：表示该层有错误，需要修正参数
    - **黄色节点**：表示该层有警告，建议优化但不影响运行
    """
    )


if __name__ == "__main__":
    # 测试运行
    architecture_designer_tab()
