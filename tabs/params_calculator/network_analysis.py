"""网络分析功能

提供完整网络的参数量、FLOPs和内存占用分析功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple

from .layer_analyzer import LayerAnalyzer


def full_network_analysis():
    """完整网络分析模式"""
    st.markdown("---")
    st.markdown("## 🏗️ 完整网络参数分析")

    st.markdown(
        """
    选择预定义网络或自定义网络架构，生成详细的参数/FLOPs报告。
    """
    )

    # 网络选择
    network_mode = st.radio(
        "选择模式", ["预定义网络", "自定义网络"], horizontal=True, key="network_mode"
    )

    if network_mode == "预定义网络":
        predefined_network_analysis()
    else:
        custom_network_analysis()

    # 返回单层分析
    if st.button("返回单层分析", use_container_width=True):
        st.session_state.calc_mode = "single"
        st.rerun()


def predefined_network_analysis():
    """预定义网络分析"""
    st.markdown("### 📦 预定义网络架构")

    network_name = st.selectbox(
        "选择网络",
        [
            "ResNet-18 (CNN)",
            "ResNet-50 (CNN)",
            "VGG-16 (CNN)",
            "MobileNetV2 (轻量级CNN)",
            "BERT-base (Transformer)",
            "GPT-2 small (Transformer)",
            "ViT-Base (Vision Transformer)",
        ],
        key="predefined_network",
    )

    # 输入尺寸
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input(
            "批次大小", min_value=1, value=1, step=1, key="batch_size"
        )
    with col2:
        input_size = st.selectbox(
            "输入尺寸", [224, 256, 384, 512], index=0, key="input_size"
        )

    # 获取网络架构
    network_config = get_network_config(network_name, input_size)

    # 计算总体统计
    total_params = 0
    total_flops = 0
    total_memory = 0

    layers_data = []

    for layer_info in network_config:
        total_params += layer_info["params"]
        total_flops += layer_info["flops"]
        total_memory += layer_info.get("memory", 0)
        layers_data.append(layer_info)

    # 显示总体统计
    st.markdown("---")
    st.markdown("### 📊 网络总体统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "总参数量", f"{total_params/1e6:.2f}M", help="网络中所有可学习参数的总数"
        )

    with col2:
        st.metric(
            "总FLOPs", f"{total_flops/1e9:.2f}G", help="单次前向传播的浮点运算次数"
        )

    with col3:
        st.metric("内存占用", f"{total_memory:.1f}MB", help="参数存储所需内存")

    with col4:
        st.metric("层数", f"{len(layers_data)}", help="网络中层的总数")

    # 参数量分布图
    st.markdown("---")
    st.markdown("### 📈 参数量分布")

    layer_names = [f"Layer {i+1}" for i in range(len(layers_data))]
    param_counts = [layer["params"] for layer in layers_data]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=layer_names,
            y=param_counts,
            marker_color="lightblue",
            text=[f"{p/1e6:.2f}M" for p in param_counts],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="各层参数量分布",
        xaxis_title="网络层",
        yaxis_title="参数量",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key="param_distribution")

    # 详细层信息表
    st.markdown("---")
    st.markdown("### 📋 详细层信息")

    # 创建详细数据表
    detailed_data = []
    for i, layer in enumerate(layers_data):
        detailed_data.append(
            {
                "层号": i + 1,
                "类型": layer["type"],
                "参数量": layer["params"],
                "FLOPs": layer["flops"],
                "输出形状": layer.get("output_shape", "-"),
            }
        )

    df = pd.DataFrame(detailed_data)
    df["参数量"] = df["参数量"].apply(
        lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.1f}K"
    )
    df["FLOPs"] = df["FLOPs"].apply(
        lambda x: f"{x/1e9:.2f}G" if x > 1e9 else f"{x/1e6:.1f}M"
    )

    st.dataframe(df, use_container_width=True)

    # 生成报告
    if st.button("📄 生成详细报告", use_container_width=True):
        generate_network_report(network_name, layers_data, total_params, total_flops)


def custom_network_analysis():
    """自定义网络分析"""
    st.markdown("### 🛠️ 自定义网络架构")

    st.markdown(
        """
    逐层构建你的网络架构，实时查看参数量和计算量。
    """
    )

    # 初始化会话状态
    if "custom_layers" not in st.session_state:
        st.session_state.custom_layers = []

    # 层配置界面
    col1, col2, col3 = st.columns(3)

    with col1:
        layer_type = st.selectbox(
            "层类型",
            [
                "Conv2d",
                "Linear",
                "BatchNorm2d",
                "LayerNorm",
                "MultiHeadAttention",
                "LSTM",
                "Embedding",
                "Dropout",
            ],
            key="layer_type",
        )

    with col2:
        if layer_type == "Conv2d":
            in_channels = st.number_input("输入通道", 3, 1024, 64, key="conv_in")
            out_channels = st.number_input("输出通道", 16, 1024, 64, key="conv_out")
            kernel_size = st.number_input("卷积核大小", 1, 11, 3, key="kernel_size")
            stride = st.number_input("步长", 1, 4, 1, key="stride")
            padding = st.number_input("填充", 0, 5, 1, key="padding")

        elif layer_type == "Linear":
            in_features = st.number_input("输入特征", 64, 4096, 512, key="linear_in")
            out_features = st.number_input("输出特征", 64, 4096, 512, key="linear_out")

        elif layer_type == "MultiHeadAttention":
            d_model = st.number_input("模型维度", 64, 2048, 512, key="d_model")
            num_heads = st.number_input("注意力头数", 1, 32, 8, key="num_heads")
            seq_len = st.number_input("序列长度", 16, 1024, 128, key="seq_len")

        elif layer_type == "LSTM":
            input_size = st.number_input("输入维度", 64, 2048, 512, key="lstm_in")
            hidden_size = st.number_input("隐藏维度", 64, 2048, 512, key="lstm_hidden")
            num_layers = st.number_input("层数", 1, 8, 2, key="lstm_layers")
            bidirectional = st.checkbox("双向", key="lstm_bidirectional")

        elif layer_type == "Embedding":
            num_embeddings = st.number_input(
                "词表大小", 1000, 100000, 10000, key="embed_vocab"
            )
            embedding_dim = st.number_input("嵌入维度", 64, 1024, 512, key="embed_dim")

    with col3:
        if st.button("➕ 添加层", use_container_width=True):
            # 构建层配置
            layer_config = {"type": layer_type}

            if layer_type == "Conv2d":
                layer_config.update(
                    {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": padding,
                    }
                )
            elif layer_type == "Linear":
                layer_config.update(
                    {
                        "in_features": in_features,
                        "out_features": out_features,
                    }
                )
            elif layer_type == "MultiHeadAttention":
                layer_config.update(
                    {
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "seq_len": seq_len,
                    }
                )
            elif layer_type == "LSTM":
                layer_config.update(
                    {
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "bidirectional": bidirectional,
                    }
                )
            elif layer_type == "Embedding":
                layer_config.update(
                    {
                        "num_embeddings": num_embeddings,
                        "embedding_dim": embedding_dim,
                    }
                )

            st.session_state.custom_layers.append(layer_config)
            st.rerun()

    # 显示已添加的层
    if st.session_state.custom_layers:
        st.markdown("---")
        st.markdown("### 📋 当前网络架构")

        # 计算各层参数量
        layers_data = []
        total_params = 0
        total_flops = 0

        for i, layer in enumerate(st.session_state.custom_layers):
            layer_info = analyze_layer(layer)
            layers_data.append(layer_info)
            total_params += layer_info["params"]
            total_flops += layer_info["flops"]

        # 显示层信息表
        df_data = []
        for i, layer in enumerate(layers_data):
            df_data.append(
                {
                    "层号": i + 1,
                    "类型": layer["type"],
                    "参数量": layer["params"],
                    "FLOPs": layer["flops"],
                }
            )

        df = pd.DataFrame(df_data)
        df["参数量"] = df["参数量"].apply(
            lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.1f}K"
        )
        df["FLOPs"] = df["FLOPs"].apply(
            lambda x: f"{x/1e9:.2f}G" if x > 1e9 else f"{x/1e6:.1f}M"
        )

        st.dataframe(df, use_container_width=True)

        # 显示总体统计
        st.markdown("### 📊 网络统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总参数量", f"{total_params/1e6:.2f}M")
        with col2:
            st.metric("总FLOPs", f"{total_flops/1e9:.2f}G")
        with col3:
            st.metric("层数", len(layers_data))

        # 清空按钮
        if st.button("🗑️ 清空所有层", use_container_width=True):
            st.session_state.custom_layers = []
            st.rerun()


def analyze_layer(layer_config: Dict) -> Dict:
    """分析单个层的参数量和FLOPs"""
    layer_type = layer_config["type"]

    if layer_type == "Conv2d":
        # 假设输入形状
        input_shape = (layer_config["in_channels"], 224, 224)
        result = LayerAnalyzer.conv2d_analysis(
            layer_config["in_channels"],
            layer_config["out_channels"],
            layer_config["kernel_size"],
            layer_config["stride"],
            layer_config["padding"],
            input_shape,
        )

    elif layer_type == "Linear":
        result = LayerAnalyzer.linear_analysis(
            layer_config["in_features"], layer_config["out_features"]
        )

    elif layer_type == "MultiHeadAttention":
        result = LayerAnalyzer.attention_analysis(
            layer_config["d_model"], layer_config["num_heads"], layer_config["seq_len"]
        )

    elif layer_type == "LSTM":
        result = LayerAnalyzer.lstm_analysis(
            layer_config["input_size"],
            layer_config["hidden_size"],
            layer_config["num_layers"],
            bidirectional=layer_config.get("bidirectional", False),
        )

    elif layer_type == "Embedding":
        result = LayerAnalyzer.embedding_analysis(
            layer_config["num_embeddings"], layer_config["embedding_dim"]
        )

    else:
        # 其他层的默认处理
        result = {
            "type": layer_type,
            "params": 0,
            "flops": 0,
            "memory": 0,
        }

    return {
        "type": layer_type,
        "params": result["parameters"]["total"],
        "flops": result["flops"]["total"],
        "memory": result.get("memory_mb", {}).get("parameters", 0),
        "output_shape": result.get("output_shape", "-"),
    }


def get_network_config(network_name: str, input_size: int) -> List[Dict]:
    """获取预定义网络配置"""
    # 这里简化实现，实际应该从模板文件加载
    if "ResNet-18" in network_name:
        return [
            {"type": "Conv2d", "params": 9408, "flops": 118013952, "memory": 0.04},
            {"type": "Conv2d", "params": 36928, "flops": 463622144, "memory": 0.14},
            {"type": "Conv2d", "params": 73856, "flops": 927244288, "memory": 0.28},
            # ... 更多层
        ]
    elif "BERT-base" in network_name:
        return [
            {"type": "Embedding", "params": 30522 * 768, "flops": 0, "memory": 89.0},
            {
                "type": "MultiHeadAttention",
                "params": 2364416,
                "flops": 2364416,
                "memory": 9.0,
            },
            # ... 更多层
        ]
    else:
        return []


def generate_network_report(
    network_name: str, layers_data: List[Dict], total_params: int, total_flops: int
):
    """生成网络分析报告"""
    st.markdown("---")
    st.markdown("### 📄 网络分析报告")

    report = f"""
    # {network_name} 网络分析报告

    ## 总体统计
    - **总参数量**: {total_params:,} ({total_params/1e6:.2f}M)
    - **总FLOPs**: {total_flops:,} ({total_flops/1e9:.2f}G)
    - **层数**: {len(layers_data)}

    ## 详细层信息
    """

    for i, layer in enumerate(layers_data):
        report += f"""
        ### Layer {i+1}: {layer['type']}
        - 参数量: {layer['params']:,}
        - FLOPs: {layer['flops']:,}
        """

    st.markdown(report)

    # 下载按钮
    st.download_button(
        label="📥 下载报告",
        data=report,
        file_name=f"{network_name}_analysis.md",
        mime="text/markdown",
    )
