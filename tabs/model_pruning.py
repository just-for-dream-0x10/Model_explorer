"""
模型剪枝计算解剖标签页
Model Pruning Computational Analysis Tab

深入解剖剪枝对神经网络数值计算的影响
核心理念：让你看到剪枝后每一步的数值变化，为什么某些参数可以安全移除
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple, Optional
import copy

from utils.memory_analyzer import get_tensor_memory


def calculate_parameter_importance(model, dataloader=None, num_samples=100):
    """
    计算参数重要性（基于梯度或激活值）

    Args:
        model: PyTorch模型
        dataloader: 数据加载器（可选）
        num_samples: 样本数量

    Returns:
        dict: 每层参数的重要性分数
    """
    importance_scores = {}

    # 简化版本：基于参数绝对值的大小
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 使用参数绝对值作为重要性指标
            importance = torch.abs(param.data).cpu().numpy()
            importance_scores[name] = importance

    return importance_scores


def structured_prune_layer(layer, pruning_ratio, method="auto"):
    """
    结构化剪枝：支持多种层类型的剪枝

    Args:
        layer: 神经网络层
        pruning_ratio: 剪枝比例
        method: 'auto', 'filter', 'channel', 'neuron'

    Returns:
        剪枝后的层和剪枝信息
    """
    original_weight = layer.weight.data.clone()
    original_bias = layer.bias.data.clone() if layer.bias is not None else None

    if isinstance(layer, nn.Conv2d):
        # 卷积层剪枝
        if method == "auto":
            method = "filter"  # 默认剪枝过滤器

        if method == "filter":
            out_channels = layer.out_channels
            filter_importance = torch.norm(
                original_weight.view(out_channels, -1), dim=1
            )
            num_filters_to_prune = int(pruning_ratio * out_channels)

            if num_filters_to_prune > 0:
                _, indices_to_prune = torch.topk(
                    filter_importance, num_filters_to_prune, largest=False
                )
                mask = torch.ones(out_channels, dtype=torch.bool)
                mask[indices_to_prune] = False
                new_out_channels = mask.sum().item()
                new_weight = original_weight[mask]
                new_bias = original_bias[mask] if original_bias is not None else None

                pruned_layer = nn.Conv2d(
                    layer.in_channels,
                    new_out_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias is not None,
                )
                pruned_layer.weight.data = new_weight
                if new_bias is not None:
                    pruned_layer.bias.data = new_bias

                return pruned_layer, {
                    "method": "filter",
                    "pruned_count": num_filters_to_prune,
                    "remaining_count": new_out_channels,
                    "indices_pruned": indices_to_prune.tolist(),
                    "original_shape": original_weight.shape,
                    "new_shape": new_weight.shape,
                    "type": "conv2d",
                }

    elif isinstance(layer, nn.Linear):
        # 全连接层剪枝
        if method == "auto":
            method = "neuron"  # 默认剪枝神经元

        if method == "neuron":
            # 剪枝输出神经元
            out_features = layer.out_features
            neuron_importance = torch.norm(original_weight, dim=1)
            num_neurons_to_prune = int(pruning_ratio * out_features)

            if num_neurons_to_prune > 0:
                _, indices_to_prune = torch.topk(
                    neuron_importance, num_neurons_to_prune, largest=False
                )
                mask = torch.ones(out_features, dtype=torch.bool)
                mask[indices_to_prune] = False
                new_out_features = mask.sum().item()
                new_weight = original_weight[mask]
                new_bias = original_bias[mask] if original_bias is not None else None

                pruned_layer = nn.Linear(
                    layer.in_features, new_out_features, bias=layer.bias is not None
                )
                pruned_layer.weight.data = new_weight
                if new_bias is not None:
                    pruned_layer.bias.data = new_bias

                return pruned_layer, {
                    "method": "neuron",
                    "pruned_count": num_neurons_to_prune,
                    "remaining_count": new_out_features,
                    "indices_pruned": indices_to_prune.tolist(),
                    "original_shape": original_weight.shape,
                    "new_shape": new_weight.shape,
                    "type": "linear",
                }

    # 不支持的层类型或剪枝方法
    return layer, {"method": method, "pruned_count": 0, "type": type(layer).__name__}


def unstructured_prune_layer(layer, pruning_ratio, method="magnitude"):
    """
    非结构化剪枝：剪枝单个参数

    Args:
        layer: 神经网络层
        pruning_ratio: 剪枝比例
        method: 'magnitude' 或 'random'

    Returns:
        剪枝后的层和掩码
    """
    original_weight = layer.weight.data.clone()
    original_bias = layer.bias.data.clone() if layer.bias is not None else None

    if method == "magnitude":
        # 基于参数绝对值大小
        weight_flat = original_weight.view(-1)
        num_params_to_prune = int(pruning_ratio * len(weight_flat))

        if num_params_to_prune > 0:
            # 选择绝对值最小的参数
            _, indices_to_prune = torch.topk(
                torch.abs(weight_flat), num_params_to_prune, largest=False
            )

            # 创建掩码
            mask = torch.ones_like(weight_flat)
            mask[indices_to_prune] = 0
            mask = mask.view(original_weight.shape)

            # 应用剪枝
            pruned_weight = original_weight * mask
            layer.weight.data = pruned_weight

            return layer, mask

    return layer, None


def analyze_pruning_impact(original_model, pruned_model, input_shape, num_samples=10):
    """
    分析剪枝对模型性能的影响

    Args:
        original_model: 原始模型
        pruned_model: 剪枝后模型
        input_shape: 输入形状
        num_samples: 测试样本数

    Returns:
        dict: 性能对比分析
    """
    # 参数量对比
    original_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())

    # 内存占用对比
    original_memory = original_params * 4 / (1024**2)  # MB
    pruned_memory = pruned_params * 4 / (1024**2)  # MB

    # 推理性能测试
    original_model.eval()
    pruned_model.eval()

    inference_times_original = []
    inference_times_pruned = []
    inference_success = True

    with torch.no_grad():
        for _ in range(num_samples):
            try:
                test_input = torch.randn(input_shape)

                # 原始模型推理时间
                if torch.cuda.is_available():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output_orig = original_model(test_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times_original.append(start_time.elapsed_time(end_time))
                else:
                    import time

                    start = time.time()
                    output_orig = original_model(test_input)
                    end = time.time()
                    inference_times_original.append((end - start) * 1000)  # ms

                # 剪枝模型推理时间
                if torch.cuda.is_available():
                    start_time.record()
                    output_pruned = pruned_model(test_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times_pruned.append(start_time.elapsed_time(end_time))
                else:
                    start = time.time()
                    output_pruned = pruned_model(test_input)
                    end = time.time()
                    inference_times_pruned.append((end - start) * 1000)

            except Exception as e:
                inference_success = False
                break

    # 输出差异分析
    mse_diff = 0.0
    cosine_sim = 1.0

    if inference_success:
        try:
            with torch.no_grad():
                test_input = torch.randn(input_shape)
                output_orig = original_model(test_input)
                output_pruned = pruned_model(test_input)

                # 计算输出差异
                mse_diff = F.mse_loss(output_orig, output_pruned).item()
                cosine_sim = F.cosine_similarity(
                    output_orig.flatten(), output_pruned.flatten(), dim=0
                ).item()
        except Exception as e:
            pass

    return {
        "parameter_reduction": {
            "original": original_params,
            "pruned": pruned_params,
            "reduction_ratio": (
                (original_params - pruned_params) / original_params
                if original_params > 0
                else 0
            ),
            "saved_params": original_params - pruned_params,
        },
        "memory_reduction": {
            "original_mb": original_memory,
            "pruned_mb": pruned_memory,
            "reduction_ratio": (
                (original_memory - pruned_memory) / original_memory
                if original_memory > 0
                else 0
            ),
            "saved_mb": original_memory - pruned_memory,
        },
        "inference_performance": {
            "original_avg_time": (
                np.mean(inference_times_original) if inference_times_original else 0
            ),
            "pruned_avg_time": (
                np.mean(inference_times_pruned) if inference_times_pruned else 0
            ),
            "speedup_ratio": (
                (np.mean(inference_times_original) / np.mean(inference_times_pruned))
                if inference_times_original
                and inference_times_pruned
                and np.mean(inference_times_pruned) > 0
                else 1.0
            ),
            "success": inference_success,
        },
        "output_similarity": {
            "mse_difference": mse_diff,
            "cosine_similarity": cosine_sim,
        },
    }


def visualize_pruning_results(importance_scores, pruning_info=None):
    """可视化剪枝结果"""
    if not importance_scores:
        return None

    # 参数重要性分布
    all_importances = []
    layer_names = []

    for name, importance in importance_scores.items():
        all_importances.extend(importance.flatten())
        layer_names.extend([name] * len(importance.flatten()))

    fig = go.Figure()

    # 添加重要性直方图
    fig.add_trace(
        go.Histogram(
            x=all_importances,
            nbinsx=50,
            name="参数重要性分布",
            marker_color="lightblue",
        )
    )

    fig.update_layout(
        title="参数重要性分布",
        xaxis_title="重要性分数",
        yaxis_title="参数数量",
        height=400,
    )

    return fig


def visualize_layer_pruning(layer_name, weight_data, mask=None):
    """可视化单层剪枝效果"""
    if len(weight_data.shape) == 4:  # Conv2d
        # 显示第一个卷积核
        kernel_data = weight_data[0, 0].cpu().numpy()

        fig = go.Figure(data=go.Heatmap(z=kernel_data, colorscale="RdBu", zmid=0))

        title = f"{layer_name} - 第一个卷积核"
        if mask is not None:
            title += " (剪枝后)"

        fig.update_layout(title=title, height=300)

        return fig

    elif len(weight_data.shape) == 2:  # Linear
        weight_matrix = weight_data.cpu().numpy()

        fig = go.Figure(data=go.Heatmap(z=weight_matrix, colorscale="RdBu", zmid=0))

        title = f"{layer_name} - 权重矩阵"
        if mask is not None:
            title += " (剪枝后)"

        fig.update_layout(title=title, height=400)

        return fig

    return None


def explain_pruning_computation():
    """解释剪枝对数值计算的影响"""
    st.markdown(
        """
    ### ✂️ 剪枝对数值计算的影响详解
    
    #### 核心问题：为什么某些参数可以安全移除？
    
    **1. 参数重要性分析**
    ```
    # 基于权重绝对值的重要性评估
    importance(i,j) = |W[i,j]|
    
    # 基于梯度的重要性评估  
    importance(i,j) = |∂L/∂W[i,j]|
    
    # 基于激活值的重要性评估
    importance(i,j) = mean(|activation[i]|)
    
    数值例子：
    原始权重矩阵 W = [[0.1, -0.8], [0.05, 0.2], [1.2, -0.3]]
    绝对值重要性 = [0.1, 0.8, 0.05, 0.2, 1.2, 0.3]
    
    排序后：[1.2, 0.8, 0.3, 0.2, 0.1, 0.05]
    剪枝50%：移除 [0.1, 0.05, 0.2]
    保留权重：[[0, -0.8], [0, 0.2], [1.2, -0.3]]
    ```
    
    **2. 剪枝后的数值变化**
    ```
    # 剪枝前：y = W · x + b
    # 剪枝后：y' = W' · x + b
    
    数值影响分析：
    x = [0.5, 0.3]
    W = [[0.1, -0.8], [0.05, 0.2]]
    b = [0.1, -0.1]
    
    剪枝前输出：
    y[0] = 0.1×0.5 + (-0.8)×0.3 + 0.1 = 0.05 - 0.24 + 0.1 = -0.09
    y[1] = 0.05×0.5 + 0.2×0.3 - 0.1 = 0.025 + 0.06 - 0.1 = -0.015
    
    剪枝后（移除小权重）：
    W' = [[0, -0.8], [0, 0.2]]
    y'[0] = 0×0.5 + (-0.8)×0.3 + 0.1 = -0.24 + 0.1 = -0.14
    y'[1] = 0×0.5 + 0.2×0.3 - 0.1 = 0.06 - 0.1 = -0.04
    
    数值变化：Δy = y' - y = [-0.05, -0.025]
    相对误差：|Δy|/|y| = [55.6%, 166.7%]
    ```
    
    **3. 梯度传播的数值影响**
    ```
    # 剪枝前梯度计算
    ∂L/∂W[i,j] = ∂L/∂y[i] × x[j]
    
    # 剪枝后梯度计算
    ∂L/∂W'[i,j] = ∂L/∂y'[i] × x[j] (如果W'[i,j] ≠ 0)
    ∂L/∂W'[i,j] = 0 (如果W'[i,j] = 0)
    
    关键洞察：
    - 被剪枝的参数不再获得梯度更新
    - 剩余参数的梯度可能发生变化
    - 可能导致梯度流不稳定
    ```
    
    **4. 数值稳定性问题**
    
    **梯度消失**：
    - 问题：剪枝后梯度变小
    - 现象：∂L/∂W' ≈ 0
    - 原因：重要连接被移除
    
    **激活值偏移**：
    - 问题：输出分布发生变化
    - 现象：y'的均值/方差与y不同
    - 影响：后续层的输入分布改变
    
    **数值精度累积**：
    - 问题：多次剪枝导致精度损失累积
    - 现象：最终输出误差逐渐增大
    - 解决：限制剪枝比例、微调恢复
    """
    )


def create_sample_model(model_type="cnn"):
    if model_type == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    elif model_type == "mlp":
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    elif model_type == "resnet_like":
        # 简化的ResNet风格
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    elif model_type == "transformer_like":
        # 简化的Transformer风格
        return nn.Sequential(
            nn.Linear(512, 512),  # Self-attention模拟
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # FFN第一层
            nn.ReLU(),
            nn.Linear(256, 512),  # FFN第二层
            nn.LayerNorm(512),
            nn.Linear(512, 10),  # 分类头
        )


def model_pruning_tab(chinese_supported=True):
    """模型剪枝分析主函数"""

    st.header("✂️ 模型剪枝计算解剖台")
    st.markdown(
        """
    > **核心理念**：深入解剖剪枝对数值计算的影响
    
    **关键问题**：
    - 剪枝后权重矩阵的数值如何变化？
    - 为什么某些参数的梯度接近零？
    - 剪枝如何影响激活值的分布？
    - 数值精度如何影响剪枝决策？
    """
    )

    st.markdown("---")

    # 剪枝计算过程解析
    with st.expander("✂️ 剪枝数值计算影响（点击展开）", expanded=False):
        explain_pruning_computation()

    st.markdown("---")

    # 分析模式选择
    st.subheader("🔧 选择分析模式")

    analysis_mode = st.radio(
        "分析模式",
        ["参数重要性分析", "结构化剪枝", "非结构化剪枝", "剪枝效果对比"],
        horizontal=True,
    )

    if analysis_mode == "参数重要性分析":
        st.markdown("---")
        st.subheader("📊 参数重要性分析")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "模型类型",
                ["cnn", "mlp", "resnet_like", "transformer_like"],
                key="importance_model_type",
            )
            importance_method = st.selectbox(
                "重要性计算方法",
                ["magnitude", "gradient", "activation"],
                key="importance_method",
            )

        with col2:
            if model_type in ["cnn", "resnet_like"]:
                input_size = st.number_input(
                    "图像尺寸", 32, 224, 32, key="importance_input_size"
                )
                input_shape_desc = f"(1, 3, {input_size}, {input_size})"
            else:  # mlp, transformer_like
                input_size = st.number_input(
                    "向量维度", 256, 1024, 512, key="importance_input_size"
                )
                input_shape_desc = f"(1, {input_size})"

            num_samples = st.number_input(
                "分析样本数", 10, 100, 50, key="importance_samples"
            )

            st.info(f"输入形状: {input_shape_desc}")

        if st.button("🔍 分析参数重要性", type="primary"):
            with st.spinner("分析中..."):
                # 创建模型
                model = create_sample_model(model_type)

                # 计算输入形状
                if model_type in ["cnn", "resnet_like"]:
                    input_shape = (1, 3, input_size, input_size)
                else:  # mlp, transformer_like
                    input_shape = (1, input_size)

                # 计算参数重要性
                importance_scores = calculate_parameter_importance(
                    model, num_samples=num_samples
                )

            st.success("✅ 分析完成！")

            # 重要性统计
            st.markdown("#### 📊 重要性统计")

            all_importances = []
            for name, importance in importance_scores.items():
                all_importances.extend(importance.flatten())

            all_importances = np.array(all_importances)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("总参数数", f"{len(all_importances):,}")
            with col2:
                st.metric("平均重要性", f"{np.mean(all_importances):.4f}")
            with col3:
                st.metric("重要性标准差", f"{np.std(all_importances):.4f}")
            with col4:
                st.metric("最大重要性", f"{np.max(all_importances):.4f}")

            # 可视化
            st.markdown("---")
            st.markdown("#### 📈 可视化分析")

            fig = visualize_pruning_results(importance_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 分层重要性分析
            st.markdown("#### 🔍 分层重要性分析")

            layer_data = []
            for name, importance in importance_scores.items():
                layer_data.append(
                    {
                        "层名": name,
                        "参数数量": importance.size,
                        "平均重要性": np.mean(importance),
                        "重要性标准差": np.std(importance),
                        "最小重要性": np.min(importance),
                        "最大重要性": np.max(importance),
                    }
                )

            df = pd.DataFrame(layer_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif analysis_mode == "结构化剪枝":
        st.markdown("---")
        st.subheader("🏗️ 结构化剪枝分析")

        st.info("💡 结构化剪枝会移除整个神经元/卷积核，适合实际部署加速")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "模型类型", ["cnn", "resnet_like"], key="structured_model"
            )
            pruning_method = st.selectbox(
                "剪枝方法",
                ["filter", "channel", "neuron"],
                help="filter: 剪枝整个卷积核; channel: 剪枝整个通道; neuron: 剪枝神经元",
            )

        with col2:
            pruning_ratio = st.slider(
                "剪枝比例", 0.1, 0.9, 0.5, 0.1, key="structured_ratio"
            )
            input_size = st.number_input(
                "输入尺寸", 32, 224, 32, key="structured_input"
            )

        if st.button("✂️ 执行结构化剪枝", type="primary"):
            with st.spinner("剪枝中..."):
                # 创建模型
                model = create_sample_model(model_type)
                original_model = copy.deepcopy(model)

                # 找到第一个可剪枝的层
                pruned_layer = None
                pruning_info = None

                for layer in model:
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        pruned_layer, pruning_info = structured_prune_layer(
                            layer, pruning_ratio, pruning_method
                        )
                        break

                if pruning_info.get("pruned_count", 0) > 0:
                    st.success(
                        f"✅ 剪枝完成！剪枝了 {pruning_info['pruned_count']} 个单元"
                    )

                    # 显示剪枝信息
                    st.markdown("#### 📋 剪枝信息")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("剪枝方法", pruning_method)
                    with col2:
                        st.metric("剪枝数量", pruning_info["pruned_count"])
                    with col3:
                        st.metric("剩余数量", pruning_info["remaining_count"])

                    # 形状变化
                    st.markdown(
                        f"""
                    **形状变化**：
                    - 原始形状：`{pruning_info['original_shape']}`
                    - 剪枝后形状：`{pruning_info['new_shape']}`
                    """
                    )

                    # 可视化
                    if isinstance(pruned_layer, (nn.Conv2d, nn.Linear)):
                        st.markdown("#### 📈 权重可视化")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**原始权重**")
                            original_fig = visualize_layer_pruning(
                                "原始层", layer.weight.data
                            )
                            if original_fig:
                                st.plotly_chart(original_fig, use_container_width=True)

                        with col2:
                            st.markdown("**剪枝后权重**")
                            pruned_fig = visualize_layer_pruning(
                                "剪枝后层", pruned_layer.weight.data
                            )
                            if pruned_fig:
                                st.plotly_chart(pruned_fig, use_container_width=True)
                else:
                    st.warning("没有进行剪枝，可能是因为剪枝比例过小")

    elif analysis_mode == "非结构化剪枝":
        st.markdown("---")
        st.subheader("🎯 非结构化剪枝分析")

        st.info("💡 非结构化剪枝移除单个参数，精度损失小但需要特殊硬件支持")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "模型类型", ["cnn", "mlp", "transformer_like"], key="unstructured_model"
            )
            pruning_method = st.selectbox(
                "剪枝策略", ["magnitude", "random"], key="unstructured_method"
            )

        with col2:
            pruning_ratio = st.slider(
                "剪枝比例", 0.1, 0.9, 0.5, 0.1, key="unstructured_ratio"
            )
            if model_type in ["cnn", "resnet_like"]:
                input_size = st.number_input(
                    "图像尺寸", 32, 128, 32, key="unstructured_input"
                )
            else:
                input_size = st.number_input(
                    "向量维度", 256, 1024, 512, key="unstructured_input"
                )

        if st.button("🎯 执行非结构化剪枝", type="primary"):
            with st.spinner("剪枝中..."):
                # 创建模型
                model = create_sample_model(model_type)
                original_model = copy.deepcopy(model)

                # 应用非结构化剪枝
                pruned_count = 0
                for layer in model.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        _, mask = unstructured_prune_layer(
                            layer, pruning_ratio, pruning_method
                        )
                        if mask is not None:
                            pruned_count += (mask == 0).sum().item()

                # 计算输入形状
                if model_type in ["cnn", "resnet_like"]:
                    input_shape = (1, 3, input_size, input_size)
                else:
                    input_shape = (1, input_size)

                # 分析剪枝影响
                impact_analysis = analyze_pruning_impact(
                    original_model, model, input_shape
                )

            st.success(f"✅ 剪枝完成！剪枝了 {pruned_count:,} 个参数")

            # 剪枝统计
            st.markdown("#### 📊 剪枝统计")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                reduction = impact_analysis["parameter_reduction"]["reduction_ratio"]
                st.metric("参数减少", f"{reduction:.1%}")
            with col2:
                saved = impact_analysis["parameter_reduction"]["saved_params"]
                st.metric("节省参数", f"{saved:,}")
            with col3:
                mem_reduction = impact_analysis["memory_reduction"]["reduction_ratio"]
                st.metric("内存减少", f"{mem_reduction:.1%}")
            with col4:
                if impact_analysis["inference_performance"]["success"]:
                    speedup = impact_analysis["inference_performance"]["speedup_ratio"]
                    st.metric("理论加速", f"{speedup:.2f}x")
                else:
                    st.metric("推理状态", "❌ 失败")

    else:  # 剪枝效果对比
        st.markdown("---")
        st.subheader("🔬 剪枝效果对比分析")

        st.info("💡 对比不同剪枝策略的效果，找到最适合的配置")

        # 配置面板
        col1, col2, col3 = st.columns(3)

        with col1:
            model_type = st.selectbox("模型类型", ["cnn", "mlp"], key="compare_model")
            if model_type == "cnn":
                input_size = st.number_input(
                    "图像尺寸", 32, 128, 32, key="compare_input"
                )
                input_shape_desc = f"(1, 3, {input_size}, {input_size})"
            else:
                input_size = st.number_input(
                    "向量维度", 256, 1024, 512, key="compare_input"
                )
                input_shape_desc = f"(1, {input_size})"
            st.info(f"输入形状: {input_shape_desc}")

        with col2:
            pruning_ratios = st.multiselect(
                "剪枝比例",
                [0.1, 0.3, 0.5, 0.7, 0.9],
                default=[0.3, 0.5, 0.7],
                key="compare_ratios",
            )

        with col3:
            methods = st.multiselect(
                "剪枝方法",
                ["structured", "unstructured"],
                default=["structured", "unstructured"],
                key="compare_methods",
            )

        if st.button("🚀 开始对比分析", type="primary"):
            if not pruning_ratios or not methods:
                st.error("请选择至少一个剪枝比例和一个剪枝方法")
                return

            with st.spinner("分析中..."):
                # 创建模型
                model = create_sample_model(model_type)
                original_model = copy.deepcopy(model)

                # 计算输入形状
                if model_type == "cnn":
                    input_shape = (1, 3, input_size, input_size)
                else:
                    input_shape = (1, input_size)

                # 对比结果
                comparison_results = {}

                for method in methods:
                    method_results = {}

                    for ratio in pruning_ratios:
                        test_model = copy.deepcopy(original_model)

                        if method == "structured":
                            # 结构化剪枝（简化版）
                            for i, layer in enumerate(test_model):
                                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                                    pruned_layer, _ = structured_prune_layer(
                                        layer, ratio, "auto"
                                    )
                                    test_model[i] = pruned_layer
                                    break  # 只剪枝第一个可剪枝的层

                        else:  # unstructured
                            # 非结构化剪枝
                            for layer in test_model.modules():
                                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                                    unstructured_prune_layer(layer, ratio, "magnitude")

                        # 分析影响
                        impact = analyze_pruning_impact(
                            original_model, test_model, input_shape
                        )

                        method_results[f"{ratio:.1f}"] = impact

                    comparison_results[method] = method_results

            st.success("✅ 对比分析完成！")

            # 结果表格
            st.markdown("#### 📊 对比结果总览")

            table_data = {
                "剪枝方法": [],
                "剪枝比例": [],
                "参数减少": [],
                "内存减少": [],
                "推理加速": [],
                "MSE差异": [],
                "余弦相似度": [],
            }

            for method, ratios_data in comparison_results.items():
                for ratio, impact in ratios_data.items():
                    table_data["剪枝方法"].append(method)
                    table_data["剪枝比例"].append(f"{ratio}")
                    table_data["参数减少"].append(
                        f"{impact['parameter_reduction']['reduction_ratio']:.1%}"
                    )
                    table_data["内存减少"].append(
                        f"{impact['memory_reduction']['reduction_ratio']:.1%}"
                    )

                    if impact["inference_performance"]["success"]:
                        table_data["推理加速"].append(
                            f"{impact['inference_performance']['speedup_ratio']:.2f}x"
                        )
                    else:
                        table_data["推理加速"].append("❌")

                    table_data["MSE差异"].append(
                        f"{impact['output_similarity']['mse_difference']:.6f}"
                    )
                    table_data["余弦相似度"].append(
                        f"{impact['output_similarity']['cosine_similarity']:.6f}"
                    )

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # 总结
    st.markdown("---")
    st.subheader("💡 剪枝最佳实践")

    st.markdown(
        """
    ### 🎯 剪枝策略选择
    
    1. **结构化剪枝**
       - 适用：移动端部署、实时推理
       - 优点：硬件友好、实际加速
       - 缺点：精度损失较大
    
    2. **非结构化剪枝**
       - 适用：研究实验、精度敏感场景
       - 优点：精度损失小、灵活度高
       - 缺点：需要特殊硬件支持
    
    ### ⚠️ 常见陷阱
    
    - **过度剪枝**：剪枝比例过高导致严重精度损失
    - **不均匀剪枝**：某些层剪枝过多破坏网络结构
    - **缺乏微调**：剪枝后不进行恢复训练
    - **忽略硬件限制**：选择不兼容的剪枝方法
    
    ### 🔧 优化建议
    
    - **渐进式剪枝**：分阶段逐步增加剪枝比例
    - **分层策略**：不同层采用不同剪枝比例
    - **微调恢复**：剪枝后进行短时间训练
    - **验证评估**：在验证集上测试剪枝效果
    """
    )


if __name__ == "__main__":
    # 测试运行
    model_pruning_tab()
