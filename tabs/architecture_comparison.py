"""
架构对比实验室
Architecture Comparison Lab

对比CNN和Transformer在不同场景下的表现
核心理念：用真实数据验证理论，提供决策依据
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.model_comparison import (
    get_model_info,
    generate_training_curves,
    compare_convergence_speed,
    get_data_efficiency_curve,
    get_comparison_recommendations,
)


def plot_training_curves(curves_dict, metric="loss"):
    """
    绘制训练曲线对比

    Args:
        curves_dict: 模型名 -> 训练曲线的字典
        metric: "loss" 或 "accuracy"

    Returns:
        fig: Plotly图表
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("训练集", "验证集"))

    colors = ["red", "blue", "green", "orange", "purple"]

    for idx, (model_name, curves) in enumerate(curves_dict.items()):
        color = colors[idx % len(colors)]
        epochs = curves["epochs"]

        if metric == "loss":
            train_data = curves["train_loss"]
            val_data = curves["val_loss"]
            ylabel = "Loss"
        else:  # accuracy
            train_data = curves["train_acc"]
            val_data = curves["val_acc"]
            ylabel = "Accuracy"

        # 训练集曲线
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_data,
                mode="lines",
                name=model_name,
                line=dict(color=color, width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # 验证集曲线
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_data,
                mode="lines",
                name=model_name,
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=2)

    fig.update_layout(
        title=f"{ylabel}曲线对比（实线=训练集，虚线=验证集）",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_model_comparison_bars(models_info):
    """
    绘制模型参数量、FLOPs对比柱状图

    Args:
        models_info: 模型信息字典列表

    Returns:
        fig: Plotly图表
    """
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("参数量 (Million)", "FLOPs (GFLOPs)")
    )

    model_names = list(models_info.keys())
    params = [info["params"] for info in models_info.values()]
    flops = [info["flops"] for info in models_info.values()]
    colors = [
        "red" if info["type"] == "CNN" else "blue" for info in models_info.values()
    ]

    # 参数量
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=params,
            text=[f"{p:.1f}M" for p in params],
            textposition="auto",
            marker_color=colors,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # FLOPs
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=flops,
            text=[f"{f:.1f}G" for f in flops],
            textposition="auto",
            marker_color=colors,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="模型复杂度对比（红色=CNN，蓝色=Transformer）", height=400)

    return fig


def plot_convergence_comparison(comparison):
    """
    绘制收敛速度对比

    Args:
        comparison: 收敛速度对比结果

    Returns:
        fig: Plotly图表
    """
    model_names = list(comparison.keys())
    epoch_90 = [stats["epoch_to_90"] for stats in comparison.values()]
    epoch_95 = [stats["epoch_to_95"] for stats in comparison.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=epoch_90,
            name="达到90%最佳精度",
            text=[f"{e}" for e in epoch_90],
            textposition="auto",
            marker_color="lightblue",
        )
    )

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=epoch_95,
            name="达到95%最佳精度",
            text=[f"{e}" for e in epoch_95],
            textposition="auto",
            marker_color="darkblue",
        )
    )

    fig.update_layout(
        title="收敛速度对比（需要多少epoch）",
        xaxis_title="模型",
        yaxis_title="Epoch数",
        barmode="group",
        height=400,
    )

    return fig


def plot_data_efficiency(cnn_curve, transformer_curve):
    """
    绘制数据效率曲线

    Args:
        cnn_curve: CNN的数据效率曲线
        transformer_curve: Transformer的数据效率曲线

    Returns:
        fig: Plotly图表
    """
    fig = go.Figure()

    # CNN曲线
    fig.add_trace(
        go.Scatter(
            x=[r * 100 for r in cnn_curve["data_ratios"]],
            y=cnn_curve["accuracies"],
            mode="lines+markers",
            name="CNN",
            line=dict(color="red", width=2),
            marker=dict(size=8),
        )
    )

    # Transformer曲线
    fig.add_trace(
        go.Scatter(
            x=[r * 100 for r in transformer_curve["data_ratios"]],
            y=transformer_curve["accuracies"],
            mode="lines+markers",
            name="Transformer",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="数据效率对比（不同数据量下的性能）",
        xaxis_title="数据量 (%)",
        yaxis_title="验证精度",
        height=400,
        hovermode="x unified",
    )

    return fig


def explain_comparison_principles():
    """解释对比实验的原理"""
    st.markdown(
        """
    ### 🔬 架构对比实验的核心问题
    
    #### 1. CNN vs Transformer：谁更好？
    
    **答案：取决于场景！**
    
    | 场景 | 推荐架构 | 原因 |
    |------|---------|------|
    | **小数据集（<10K）** | CNN | 归纳偏置强，泛化能力好 |
    | **中等数据集（10K-100K）** | CNN | 训练更稳定，效果相当 |
    | **大数据集（>100K）** | CNN或ViT | 都可以，ViT可能略好 |
    | **超大数据集（>1M）** | ViT | 数据充足时ViT优势明显 |
    | **边缘设备** | 轻量级CNN | 参数少，推理快 |
    | **云端部署** | ViT | 可以利用大模型优势 |
    
    #### 2. 归纳偏置（Inductive Bias）的影响
    
    **CNN的归纳偏置**：
    - ✅ **平移不变性**：卷积核在图像上滑动，对目标位置不敏感
    - ✅ **局部性**：卷积关注局部区域，符合图像的空间结构
    - ✅ **参数共享**：同一个卷积核用于所有位置
    
    **结果**：小数据集上CNN表现好，因为"内置"了图像的先验知识
    
    **ViT的归纳偏置**：
    - ❌ **无平移不变性**：需要位置编码来区分位置
    - ❌ **无局部性假设**：Self-Attention一开始就看全局
    - ❌ **无参数共享**：每个位置的参数都不同
    
    **结果**：需要大量数据来学习这些模式，但学到后可能更灵活
    
    #### 3. 数据效率的数学解释
    
    **泛化误差分解**：
    ```
    泛化误差 = 偏差² + 方差 + 噪声
    ```
    
    **CNN**：
    - 高偏差（强假设限制了模型能力）
    - 低方差（稳定，不容易过拟合）
    - → 小数据集上稳定
    
    **ViT**：
    - 低偏差（灵活，表达能力强）
    - 高方差（容易过拟合小数据集）
    - → 大数据集上表现好
    
    #### 4. 实验设计原则
    
    **本实验采用的对比维度**：
    1. **训练曲线**：Loss和Accuracy随epoch的变化
    2. **收敛速度**：达到目标精度需要多少epoch
    3. **最终性能**：最佳验证精度
    4. **数据效率**：不同数据量下的性能
    5. **计算成本**：参数量、FLOPs
    
    **为什么不做实时训练**：
    - 保持交互性（用户不需要等待）
    - 使用基于真实实验规律的模拟数据
    - 结论仍然有效且具有教学价值
    """
    )


def architecture_comparison_tab(chinese_supported=True):
    """架构对比实验室主函数"""

    st.header("🔬 架构对比实验室")
    st.markdown(
        """
    > **核心问题**：CNN vs Transformer，什么时候用哪个？用数据说话！
    
    **实验方法**：对比训练曲线、收敛速度、数据效率、计算成本
    """
    )

    st.markdown("---")

    # 实验原理
    with st.expander("🔬 实验原理（点击展开）", expanded=False):
        explain_comparison_principles()

    st.markdown("---")

    # 实验配置
    st.subheader("⚙️ 实验配置")

    col1, col2 = st.columns(2)

    with col1:
        dataset_size = st.selectbox(
            "数据集规模",
            ["small", "medium", "large"],
            format_func=lambda x: {
                "small": "小数据集 (~10K图像)",
                "medium": "中等数据集 (~50K图像)",
                "large": "大数据集 (~500K图像)",
            }[x],
        )

    with col2:
        num_epochs = st.slider("训练轮数", 20, 200, 100, 10)

    # 模型选择
    st.markdown("#### 选择对比的模型")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**CNN模型**")
        cnn_models = st.multiselect(
            "CNN",
            ["ResNet-18", "ResNet-50", "MobileNet-V2"],
            default=["ResNet-18"],
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("**Transformer模型**")
        vit_models = st.multiselect(
            "ViT",
            ["ViT-Tiny", "ViT-Small", "ViT-Base"],
            default=["ViT-Tiny"],
            label_visibility="collapsed",
        )

    with col3:
        st.markdown("**对比维度**")
        show_loss = st.checkbox("Loss曲线", value=True)
        show_acc = st.checkbox("Accuracy曲线", value=True)
        show_convergence = st.checkbox("收敛速度", value=True)

    selected_models = cnn_models + vit_models

    if len(selected_models) == 0:
        st.warning("⚠️ 请至少选择一个模型")
        return

    if len(selected_models) > 4:
        st.warning("⚠️ 最多选择4个模型进行对比")
        selected_models = selected_models[:4]

    # 运行实验
    st.markdown("---")
    st.subheader("📊 实验结果")

    if st.button("🚀 运行对比实验", type="primary"):
        with st.spinner("生成实验数据..."):
            # 获取模型信息
            models_info = {}
            for model_name in selected_models:
                models_info[model_name] = get_model_info(model_name)

            # 生成训练曲线
            curves_dict = {}
            for model_name, info in models_info.items():
                model_type = info["type"]
                curves = generate_training_curves(
                    model_type, dataset_size, num_epochs=num_epochs
                )
                curves_dict[model_name] = curves

        st.success("✅ 实验完成！")

        # 显示模型信息对比
        st.markdown("#### 📋 模型基本信息")

        # 创建对比表格
        comparison_data = {
            "模型": [],
            "类型": [],
            "参数量(M)": [],
            "FLOPs(G)": [],
            "深度": [],
            "归纳偏置": [],
        }

        for model_name, info in models_info.items():
            comparison_data["模型"].append(model_name)
            comparison_data["类型"].append(info["type"])
            comparison_data["参数量(M)"].append(f"{info['params']:.1f}")
            comparison_data["FLOPs(G)"].append(f"{info['flops']:.1f}")
            comparison_data["深度"].append(info["depth"])
            comparison_data["归纳偏置"].append(info["inductive_bias"])

        st.table(comparison_data)

        # 参数量和FLOPs对比图
        fig1 = plot_model_comparison_bars(models_info)
        st.plotly_chart(fig1, use_container_width=True)

        # Loss曲线
        if show_loss:
            st.markdown("---")
            st.markdown("#### 📈 Loss曲线对比")
            fig2 = plot_training_curves(curves_dict, metric="loss")
            st.plotly_chart(fig2, use_container_width=True)

            st.info(
                """
            **观察要点**：
            - 实线 = 训练集，虚线 = 验证集
            - 验证集Loss高于训练集 → 可能过拟合
            - Loss下降速度 → 收敛快慢
            """
            )

        # Accuracy曲线
        if show_acc:
            st.markdown("---")
            st.markdown("#### 📈 Accuracy曲线对比")
            fig3 = plot_training_curves(curves_dict, metric="accuracy")
            st.plotly_chart(fig3, use_container_width=True)

            # 显示最终精度
            st.markdown("**最终验证精度对比**：")
            for model_name, curves in curves_dict.items():
                final_acc = curves["final_val_acc"]
                best_acc = curves["best_val_acc"]
                st.write(
                    f"- **{model_name}**: 最终={final_acc:.4f}, 最佳={best_acc:.4f}"
                )

        # 收敛速度对比
        if show_convergence:
            st.markdown("---")
            st.markdown("#### ⚡ 收敛速度对比")

            comparison = compare_convergence_speed(curves_dict)
            fig4 = plot_convergence_comparison(comparison)
            st.plotly_chart(fig4, use_container_width=True)

            st.info(
                """
            **解读**：
            - 柱子越短 = 收敛越快
            - 对比"达到90%最佳精度"和"达到95%最佳精度"
            - CNN通常在小数据集上收敛更快
            """
            )

        # 数据集规模影响分析
        st.markdown("---")
        st.markdown("#### 📊 关键发现")

        # 根据数据集规模给出结论
        cnn_accs = [
            curves_dict[m]["best_val_acc"] for m in cnn_models if m in curves_dict
        ]
        vit_accs = [
            curves_dict[m]["best_val_acc"] for m in vit_models if m in curves_dict
        ]

        if cnn_accs and vit_accs:
            avg_cnn = np.mean(cnn_accs)
            avg_vit = np.mean(vit_accs)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("CNN平均精度", f"{avg_cnn:.4f}")
            with col2:
                st.metric("Transformer平均精度", f"{avg_vit:.4f}")
            with col3:
                diff = avg_vit - avg_cnn
                st.metric(
                    "精度差异",
                    f"{diff:.4f}",
                    delta=f"{'ViT领先' if diff > 0 else 'CNN领先'}",
                )

            # 给出结论
            if dataset_size == "small":
                if avg_cnn > avg_vit:
                    st.success(
                        f"""
                    ✅ **符合预期**：在小数据集上，CNN（{avg_cnn:.4f}）优于Transformer（{avg_vit:.4f}）
                    
                    **原因**：CNN的归纳偏置（平移不变性、局部性）在小数据集上提供了更好的泛化能力
                    """
                    )
                else:
                    st.warning("⚠️ 注意：这个结果有些意外，可能是模型配置或随机性的影响")

            elif dataset_size == "large":
                if avg_vit >= avg_cnn:
                    st.success(
                        f"""
                    ✅ **符合预期**：在大数据集上，Transformer（{avg_vit:.4f}）表现优于或接近CNN（{avg_cnn:.4f}）
                    
                    **原因**：数据充足时，Transformer的强大表达能力开始显现，弱归纳偏置反而成为优势
                    """
                    )
                else:
                    st.info(
                        "ℹ️ CNN仍然有优势，可能是因为数据量还不够大，或者任务特性更适合CNN"
                    )

            else:  # medium
                st.info(
                    f"""
                ℹ️ **中等数据集**：CNN（{avg_cnn:.4f}）vs Transformer（{avg_vit:.4f}）
                
                在中等规模数据集上，两者性能接近，选择取决于具体需求：
                - 追求稳定性 → CNN
                - 追求天花板 → Transformer（需要更多调参）
                """
                )

    # 数据效率实验
    st.markdown("---")
    st.subheader("📊 数据效率分析")

    st.markdown(
        """
    **核心问题**：不同数据量下，CNN和Transformer的表现如何？
    
    这个实验回答了"为什么ViT需要大数据集"这个经典问题。
    """
    )

    if st.button("🔬 运行数据效率实验"):
        with st.spinner("生成数据效率曲线..."):
            cnn_curve = get_data_efficiency_curve("CNN")
            transformer_curve = get_data_efficiency_curve("Transformer")

        fig5 = plot_data_efficiency(cnn_curve, transformer_curve)
        st.plotly_chart(fig5, use_container_width=True)

        st.success(
            """
        ✅ **关键发现**：
        
        1. **小数据量（10%）**：
           - CNN: ~0.65精度
           - Transformer: ~0.50精度
           - **CNN明显领先**
        
        2. **大数据量（100%）**：
           - CNN: ~0.88精度
           - Transformer: ~0.91精度
           - **Transformer追上并超越**
        
        3. **提升幅度**：
           - CNN: 0.65 → 0.88（+35%）
           - Transformer: 0.50 → 0.91（+82%）
           - **Transformer对数据更敏感**
        
        **结论**：Transformer是"数据饥渴型"模型，需要大量数据才能发挥优势
        """
        )

    # 决策助手
    st.markdown("---")
    st.subheader("🎯 模型选择决策助手")

    st.markdown("回答几个问题，获取模型推荐：")

    col1, col2, col3 = st.columns(3)

    with col1:
        user_data_size = st.selectbox(
            "你的数据集规模",
            ["small", "medium", "large"],
            format_func=lambda x: {
                "small": "小（<10K）",
                "medium": "中（10K-100K）",
                "large": "大（>100K）",
            }[x],
        )

    with col2:
        user_compute = st.selectbox(
            "计算资源",
            ["low", "medium", "high"],
            format_func=lambda x: {
                "low": "低（CPU/边缘设备）",
                "medium": "中（单GPU）",
                "high": "高（多GPU/TPU）",
            }[x],
        )

    with col3:
        user_task = st.selectbox(
            "任务类型", ["classification", "detection", "segmentation"]
        )

    if st.button("💡 获取推荐"):
        rec = get_comparison_recommendations(user_data_size, user_compute, user_task)

        st.success(
            f"""
        ### 推荐结果
        
        **首选模型**: {rec['primary']}
        
        **备选模型**: {rec['alternative']}
        
        **推荐理由**: {rec['reason']}
        """
        )

        # 显示首选模型的详细信息
        primary_info = get_model_info(rec["primary"])
        st.markdown(
            f"""
        #### {rec['primary']} 详细信息
        
        - **类型**: {primary_info['type']}
        - **参数量**: {primary_info['params']}M
        - **FLOPs**: {primary_info['flops']} GFLOPs
        - **最适合**: {primary_info['best_for']}
        - **预训练数据集**: {primary_info['pretrain_dataset']}
        """
        )

    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")

    st.markdown(
        """
    ### 实验结论总结
    
    1. **小数据集（<10K）**
       - ✅ CNN占优：归纳偏置提供强泛化能力
       - ❌ ViT表现差：容易过拟合
       - 推荐：ResNet、MobileNet
    
    2. **中等数据集（10K-100K）**
       - 🤝 CNN和ViT接近
       - CNN更稳定，ViT潜力更大
       - 推荐：ResNet-50或ViT-Small
    
    3. **大数据集（>100K）**
       - ✅ ViT占优：强大表达能力显现
       - CNN仍然不错，但天花板较低
       - 推荐：ViT-Base或ViT-Large
    
    ### 选择决策树
    
    ```
    数据量 < 10K?
    ├─ 是 → CNN（ResNet-18、MobileNet-V2）
    └─ 否 → 数据量 > 100K?
        ├─ 是 → 算力充足?
        │   ├─ 是 → ViT（ViT-Base）
        │   └─ 否 → CNN（ResNet-50）
        └─ 否 → CNN（ResNet-50）或 ViT-Small
    ```
    
    ### 记住三个关键点
    
    1. **归纳偏置 = 先验知识**
       - CNN内置了图像的先验（局部性、平移不变性）
       - ViT没有先验，需要从数据学习
    
    2. **数据效率差异巨大**
       - CNN在小数据上稳定
       - ViT需要10倍以上的数据才能发挥优势
    
    3. **没有绝对的"最好"**
       - 取决于数据量、计算资源、任务特点
       - 实验和对比是最可靠的方法
    """
    )


if __name__ == "__main__":
    # 测试运行
    architecture_comparison_tab()
