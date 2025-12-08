"""
数值稳定性诊断标签页
Numerical Stability Diagnosis Tab

自动检测神经网络的数值稳定性问题
核心理念：让你看到哪一层出了什么数值问题
"""

import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from utils.stability_analyzer import (
    check_activation_stability,
    check_gradient_stability,
    check_weight_stability,
    analyze_model_stability,
    LayerStabilityInfo,
    # Phase 3 新增
    detect_gradient_flow_realtime,
    recommend_initialization,
    predict_peak_memory,
    format_memory_size,
)


def plot_gradient_flow(layers_info):
    """绘制梯度流图"""
    layer_names = [info.name for info in layers_info]
    gradient_norms = [info.gradient_norm for info in layers_info]

    # 状态颜色
    colors = []
    for info in layers_info:
        if info.gradient_status == "梯度消失":
            colors.append("red")
        elif info.gradient_status == "梯度爆炸":
            colors.append("orange")
        elif info.gradient_status == "包含NaN":
            colors.append("purple")
        else:
            colors.append("green")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(layer_names))),
            y=gradient_norms,
            mode="lines+markers",
            name="梯度范数",
            line=dict(color="blue", width=2),
            marker=dict(size=10, color=colors, line=dict(color="black", width=1)),
            text=layer_names,
            hovertemplate="<b>%{text}</b><br>梯度范数: %{y:.2e}<extra></extra>",
        )
    )

    # 添加警戒线
    fig.add_hline(
        y=1e-7,
        line_dash="dash",
        line_color="red",
        annotation_text="梯度消失警戒线 (1e-7)",
    )
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="orange",
        annotation_text="梯度爆炸警戒线 (10)",
    )

    fig.update_layout(
        title="梯度流分析（对数坐标）",
        xaxis_title="层索引",
        yaxis_title="梯度范数（对数）",
        yaxis_type="log",
        height=500,
        showlegend=False,
    )

    return fig


def plot_activation_range(layers_info):
    """绘制激活值范围"""
    layer_names = [info.name for info in layers_info]
    act_mins = [info.activation_min for info in layers_info]
    act_maxs = [info.activation_max for info in layers_info]
    act_means = [info.activation_mean for info in layers_info]

    fig = go.Figure()

    # 激活值范围（min到max的线）
    for i, name in enumerate(layer_names):
        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[act_mins[i], act_maxs[i]],
                mode="lines",
                line=dict(color="lightgray", width=8),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 均值点
    fig.add_trace(
        go.Scatter(
            x=list(range(len(layer_names))),
            y=act_means,
            mode="markers",
            name="均值",
            marker=dict(size=10, color="blue"),
            text=layer_names,
            hovertemplate="<b>%{text}</b><br>均值: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="激活值范围分析", xaxis_title="层索引", yaxis_title="激活值", height=500
    )

    return fig


def plot_weight_distribution(layers_info):
    """绘制权重分布"""
    layer_names = [info.name for info in layers_info if info.weight_std > 0]
    weight_stds = [info.weight_std for info in layers_info if info.weight_std > 0]

    if not layer_names:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=layer_names,
            y=weight_stds,
            text=[f"{std:.4f}" for std in weight_stds],
            textposition="auto",
            marker_color="green",
        )
    )

    fig.update_layout(
        title="权重标准差分布", xaxis_title="层", yaxis_title="标准差", height=400
    )

    return fig


def explain_stability_concepts():
    """解释数值稳定性概念"""
    st.markdown(
        """
    ### 🔬 数值稳定性诊断原理
    
    #### 三大检测维度
    
    **1. 梯度稳定性**
    ```python
    梯度范数 = ||∇L/∇θ||
    
    判断标准：
    - 梯度范数 < 1e-7  → 梯度消失 ❌
    - 梯度范数 > 10    → 梯度爆炸 ⚠️
    - 包含NaN/Inf      → 训练崩溃 💥
    - 其他             → 正常 ✅
    ```
    
    **2. 激活值稳定性**
    ```python
    激活值范围 = [min, max]
    
    判断标准：
    - |max| > 100 或 |min| > 100  → 异常大 ⚠️
    - mean < 1e-3 且 std < 1e-3   → 异常小 ⚠️
    - 包含NaN/Inf                  → 数值溢出 💥
    - 其他                         → 正常 ✅
    ```
    
    **3. 权重稳定性**
    ```python
    权重标准差 = std(weights)
    
    判断标准：
    - std < 1e-6   → 未初始化或异常 ❌
    - std > 10     → 权重失控 ⚠️
    - 包含NaN/Inf  → 训练崩溃 💥
    - 其他         → 正常 ✅
    ```
    
    #### 常见问题和解决方案
    
    | 问题 | 症状 | 原因 | 解决方案 |
    |------|------|------|----------|
    | **梯度消失** | 梯度范数<1e-7 | 激活函数饱和、网络过深 | 使用ResNet、ReLU、He初始化 |
    | **梯度爆炸** | 梯度范数>10 | 权重过大、学习率过大 | 梯度裁剪、降低学习率、BatchNorm |
    | **激活值过大** | |值|>100 | 权重初始化不当 | Xavier/He初始化、BatchNorm |
    | **权重失控** | std>10 | 学习率过大、无正则化 | 降低学习率、添加weight decay |
    | **NaN/Inf** | 包含NaN/Inf | 数值溢出 | 降低学习率、梯度裁剪、检查输入 |
    
    #### 诊断流程
    
    1. **前向传播** - 收集每层的激活值
    2. **反向传播** - 收集每层的梯度
    3. **统计分析** - 计算均值、标准差、范数
    4. **问题检测** - 对比阈值，识别异常
    5. **建议生成** - 根据问题类型给出解决方案
    """
    )


def stability_diagnosis_tab(chinese_supported=True):
    """数值稳定性诊断主函数"""

    st.header("⚠️ 数值稳定性诊断")
    st.markdown(
        """
    > **核心功能**：自动检测神经网络训练时的数值稳定性问题
    
    **检测项目**：梯度消失/爆炸、激活值异常、权重异常、NaN/Inf
    """
    )

    st.markdown("---")

    # 诊断原理
    with st.expander("🔬 诊断原理（点击展开）", expanded=False):
        explain_stability_concepts()

    st.markdown("---")

    # 快速测试
    st.subheader("🧪 快速稳定性测试")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**测试场景**")
        test_scenario = st.selectbox(
            "选择测试场景",
            [
                "正常网络（5层CNN）",
                "深层网络无残差（50层）",
                "未正确初始化网络",
                "学习率过大网络",
            ],
        )

    with col2:
        st.markdown("**参数配置**")
        batch_size = st.number_input("Batch Size", 1, 16, 4, key="diag_batch")
        input_size = st.selectbox("输入尺寸", [32, 64, 224], index=1, key="diag_input")

    if st.button("🚀 开始诊断", type="primary"):
        with st.spinner("诊断中...这可能需要几秒钟"):
            # 根据场景创建模型
            if test_scenario == "正常网络（5层CNN）":
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                )

            elif test_scenario == "深层网络无残差（50层）":
                layers = []
                in_ch = 3
                for i in range(50):
                    out_ch = 64 if i > 0 else 32
                    layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                    layers.append(nn.Sigmoid())  # 容易梯度消失
                    in_ch = out_ch
                layers.extend(
                    [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
                )
                model = nn.Sequential(*layers)

            elif test_scenario == "未正确初始化网络":
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                )
                # 故意使用不当初始化
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.constant_(m.weight, 0.0)  # 全零初始化

            else:  # 学习率过大
                model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                )

            # 生成输入
            input_data = torch.randn(batch_size, 3, input_size, input_size)

            # 分析稳定性
            result = analyze_model_stability(model, input_data, num_steps=5)

        st.success("✅ 诊断完成！")

        # 显示总结
        st.markdown("#### 📊 诊断总结")

        summary = result["summary"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总层数", summary["total_layers"])
        with col2:
            st.metric(
                "问题层数",
                summary["problem_layers"],
                delta=f"{summary['total_issues']}个问题",
            )
        with col3:
            st.metric(
                "梯度消失",
                summary["gradient_vanish_count"],
                delta="层" if summary["gradient_vanish_count"] > 0 else None,
                delta_color="inverse",
            )
        with col4:
            st.metric(
                "梯度爆炸",
                summary["gradient_explode_count"],
                delta="层" if summary["gradient_explode_count"] > 0 else None,
                delta_color="inverse",
            )

        # 问题层详情
        if result["problem_layers"]:
            st.markdown("---")
            st.markdown("#### ⚠️ 问题层详细报告")

            for info in result["problem_layers"]:
                with st.expander(f"❌ {info.name} ({info.layer_type})", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**激活值状态**")
                        if info.activation_status != "正常":
                            st.error(f"状态: {info.activation_status}")
                        else:
                            st.success(f"状态: {info.activation_status}")
                        st.write(
                            f"范围: [{info.activation_min:.2f}, {info.activation_max:.2f}]"
                        )
                        st.write(f"均值: {info.activation_mean:.4f}")
                        st.write(f"标准差: {info.activation_std:.4f}")

                    with col2:
                        st.markdown("**梯度状态**")
                        if info.gradient_status == "梯度消失":
                            st.error(f"状态: {info.gradient_status}")
                        elif info.gradient_status == "梯度爆炸":
                            st.warning(f"状态: {info.gradient_status}")
                        else:
                            st.success(f"状态: {info.gradient_status}")
                        st.write(f"范数: {info.gradient_norm:.2e}")
                        st.write(f"最大值: {info.gradient_max:.2e}")

                    with col3:
                        st.markdown("**权重状态**")
                        if info.weight_status != "正常":
                            st.warning(f"状态: {info.weight_status}")
                        else:
                            st.success(f"状态: {info.weight_status}")
                        st.write(f"均值: {info.weight_mean:.4f}")
                        st.write(f"标准差: {info.weight_std:.4f}")

                    # 问题和建议
                    if info.issues:
                        st.markdown("**🔍 发现的问题：**")
                        for issue in info.issues:
                            st.error(f"• {issue}")

                    if info.recommendations:
                        st.markdown("**💡 建议的解决方案：**")
                        for rec in info.recommendations:
                            st.success(f"• {rec}")

        else:
            st.success("🎉 未检测到稳定性问题！所有层都正常。")

        # 可视化
        if result["layers"]:
            st.markdown("---")
            st.markdown("#### 📈 可视化分析")

            tab1, tab2, tab3 = st.tabs(["梯度流", "激活值范围", "权重分布"])

            with tab1:
                fig1 = plot_gradient_flow(result["layers"])
                st.plotly_chart(fig1, use_container_width=True)

                st.info(
                    """
                **图表解读**：
                - 🟢 绿色点 = 正常梯度
                - 🔴 红色点 = 梯度消失
                - 🟠 橙色点 = 梯度爆炸
                - 🟣 紫色点 = 包含NaN
                """
                )

            with tab2:
                fig2 = plot_activation_range(result["layers"])
                st.plotly_chart(fig2, use_container_width=True)

                st.info(
                    """
                **图表解读**：
                - 灰色线 = 激活值的范围（min到max）
                - 蓝色点 = 激活值的均值
                - 范围过大(>100)或过小(<0.001)可能有问题
                """
                )

            with tab3:
                fig3 = plot_weight_distribution(result["layers"])
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)

                    st.info(
                        """
                    **图表解读**：
                    - 权重标准差反映初始化质量
                    - 理想范围：0.01 - 1.0
                    - 过小(<0.001)：可能未正确初始化
                    - 过大(>10)：权重增长失控
                    """
                    )
                else:
                    st.warning("无权重数据可视化")

    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")

    st.markdown(
        """
    ### 数值稳定性的重要性
    
    1. **早期发现问题**
       - 在训练前就能发现潜在问题
       - 避免浪费时间在无法收敛的网络上
    
    2. **针对性优化**
       - 知道哪一层有问题
       - 针对性地调整那一层
    
    3. **避免常见陷阱**
       - 梯度消失/爆炸
       - 权重初始化不当
       - 学习率设置不当
    
    ### 最佳实践
    
    1. **网络设计**
       - 深度网络使用残差连接
       - 使用ReLU/GELU激活函数
       - 添加BatchNorm/LayerNorm
    
    2. **权重初始化**
       - Conv2d使用He初始化
       - Linear使用Xavier初始化
       - 避免全零或全一初始化
    
    3. **训练技巧**
       - 使用梯度裁剪（clip_grad_norm）
       - 合理设置学习率（0.001-0.01）
       - 使用学习率调度器
       - 添加权重衰减（L2正则化）
    
    ### 快速诊断清单
    
    - [ ] 梯度范数在合理范围（1e-5 到 10）
    - [ ] 激活值不会太大或太小
    - [ ] 权重标准差在合理范围
    - [ ] 没有NaN或Inf
    - [ ] 深度网络使用了残差连接或BatchNorm
    - [ ] 使用了合适的初始化方案
    """
    )
    
    # ==================== Phase 3: 新增高级功能 ====================
    st.markdown("---")
    st.markdown("## 🚀 Phase 3: 高级诊断工具")
    
    tab1, tab2, tab3 = st.tabs(["🔍 实时梯度检测", "💡 初始化推荐", "💾 内存预测"])
    
    with tab1:
        st.markdown("### 🔍 实时梯度检测")
        st.markdown("检测梯度消失、梯度爆炸和数值溢出问题")
        
        if st.button("🚀 运行梯度检测", type="primary"):
            with st.spinner("正在分析梯度流动..."):
                try:
                    # 创建测试模型
                    test_model = nn.Sequential(
                        nn.Linear(100, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
                    
                    sample_input = torch.randn(4, 100)
                    
                    # 运行梯度检测
                    gradient_info = detect_gradient_flow_realtime(test_model, sample_input)
                    
                    # 显示健康状态
                    if gradient_info['healthy']:
                        st.success("✅ 梯度流动健康！所有层的梯度都在正常范围内")
                    else:
                        st.error("⚠️ 检测到梯度问题！")
                    
                    # 显示统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    stats = gradient_info['statistics']
                    with col1:
                        st.metric("平均梯度范数", f"{stats['mean_norm']:.2e}")
                    with col2:
                        st.metric("最大梯度范数", f"{stats['max_norm']:.2e}")
                    with col3:
                        st.metric("最小梯度范数", f"{stats['min_norm']:.2e}")
                    with col4:
                        st.metric("标准差", f"{stats['std_norm']:.2e}")
                    
                    # 显示问题层
                    if gradient_info['vanishing']:
                        st.warning(f"🟡 梯度消失: {len(gradient_info['vanishing'])} 层")
                        with st.expander("查看详情"):
                            for layer, norm in gradient_info['vanishing'].items():
                                st.write(f"- {layer}: 梯度范数 = {norm:.2e}")
                    
                    if gradient_info['exploding']:
                        st.error(f"🔴 梯度爆炸: {len(gradient_info['exploding'])} 层")
                        with st.expander("查看详情"):
                            for layer, norm in gradient_info['exploding'].items():
                                st.write(f"- {layer}: 梯度范数 = {norm:.2e}")
                    
                    if gradient_info['nan_inf']:
                        st.error(f"🔴 数值溢出: {len(gradient_info['nan_inf'])} 层")
                        with st.expander("查看详情"):
                            for layer, info in gradient_info['nan_inf'].items():
                                st.write(f"- {layer}: NaN={info['has_nan']}, Inf={info['has_inf']}")
                    
                    # 显示建议
                    if gradient_info['recommendations']:
                        st.markdown("### 💡 修复建议")
                        for rec in gradient_info['recommendations']:
                            with st.expander(f"{rec['issue']} (严重性: {rec['severity']})"):
                                st.markdown("**受影响的层:**")
                                for layer in rec['affected_layers'][:5]:
                                    st.write(f"- {layer}")
                                if len(rec['affected_layers']) > 5:
                                    st.write(f"- ... 还有 {len(rec['affected_layers']) - 5} 层")
                                
                                st.markdown("**建议:**")
                                for suggestion in rec['suggestions']:
                                    st.write(f"- {suggestion}")
                
                except Exception as e:
                    st.error(f"梯度检测失败: {e}")
    
    with tab2:
        st.markdown("### 💡 初始化方案推荐")
        st.markdown("根据层类型和激活函数推荐最佳初始化方案")
        
        col1, col2 = st.columns(2)
        with col1:
            layer_type = st.selectbox(
                "选择层类型",
                ["Conv2d", "Linear", "LSTM", "BatchNorm2d"]
            )
        with col2:
            activation = st.selectbox(
                "选择激活函数",
                ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU"]
            )
        
        if st.button("🎯 获取推荐", type="primary"):
            # 创建测试层
            if layer_type == "Conv2d":
                test_layer = nn.Conv2d(3, 64, 3)
            elif layer_type == "Linear":
                test_layer = nn.Linear(100, 256)
            elif layer_type == "LSTM":
                test_layer = nn.LSTM(100, 256)
            else:
                test_layer = nn.BatchNorm2d(64)
            
            # 获取推荐
            rec = recommend_initialization(test_layer, layer_type, activation.lower())
            
            st.success(f"✅ 推荐方法: **{rec['method']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**原因:**\n\n{rec['reason']}")
            with col2:
                st.info(f"**说明:**\n\n{rec['description']}")
            
            st.markdown("### 📝 代码示例")
            st.code(rec['code'], language='python')
            
            if 'bias_init' in rec:
                st.markdown("### 偏置初始化")
                st.code(rec['bias_init']['code'], language='python')
                st.caption(rec['bias_init']['reason'])
    
    with tab3:
        st.markdown("### 💾 峰值内存预测")
        st.markdown("预测训练时的内存使用，包括参数、梯度、优化器状态和激活值")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.number_input("批大小", 1, 128, 32)
        with col2:
            optimizer = st.selectbox("优化器", ["Adam", "SGD", "AdamW"])
        with col3:
            precision = st.selectbox("精度", ["float32", "float16", "float64"])
        
        if st.button("📊 预测内存", type="primary"):
            with st.spinner("正在计算..."):
                try:
                    # 创建测试模型
                    test_model = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(128, 10)
                    )
                    
                    dtype = getattr(torch, precision)
                    
                    memory_info = predict_peak_memory(
                        test_model,
                        input_shape=(3, 224, 224),
                        batch_size=batch_size,
                        optimizer_type=optimizer.lower(),
                        dtype=dtype
                    )
                    
                    # 显示总内存
                    st.markdown(f"### 📊 预测峰值内存: **{format_memory_size(memory_info['total_peak'])}**")
                    
                    # 显示分解
                    st.markdown("#### 内存分解")
                    breakdown = memory_info['breakdown']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("参数", format_memory_size(breakdown['parameters']))
                        st.metric("梯度", format_memory_size(breakdown['gradients']))
                    with col2:
                        st.metric("优化器状态", format_memory_size(breakdown['optimizer_states']))
                        st.metric("前向激活", format_memory_size(breakdown['forward_activations']))
                    with col3:
                        st.metric("反向激活", format_memory_size(breakdown['backward_activations']))
                        st.metric("参数数量", f"{memory_info['parameter_count']:,}")
                    
                    # 显示内存对比
                    st.markdown("#### 🔄 不同配置下的内存对比")
                    comparison = memory_info['memory_comparison']
                    
                    import pandas as pd
                    df = pd.DataFrame({
                        '配置': ['当前配置', '减半批大小', '混合精度', 'SGD优化器'],
                        '内存 (MB)': [
                            comparison['current'],
                            comparison['half_batch'],
                            comparison['mixed_precision'],
                            comparison['sgd_optimizer']
                        ]
                    })
                    df['内存 (格式化)'] = df['内存 (MB)'].apply(format_memory_size)
                    df['节省'] = ((df['内存 (MB)'].iloc[0] - df['内存 (MB)']) / df['内存 (MB)'].iloc[0] * 100).round(1).astype(str) + '%'
                    
                    st.dataframe(df[['配置', '内存 (格式化)', '节省']], use_container_width=True)
                    
                    # 显示建议
                    if memory_info['recommendations']:
                        st.markdown("### 💡 优化建议")
                        for rec in memory_info['recommendations']:
                            severity_color = {
                                'info': 'info',
                                'medium': 'warning',
                                'high': 'error'
                            }.get(rec['severity'], 'info')
                            
                            with st.expander(f"{rec['issue']} ({rec['severity']})"):
                                for suggestion in rec['suggestions']:
                                    st.write(f"- {suggestion}")
                
                except Exception as e:
                    st.error(f"内存预测失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    # 测试运行
    stability_diagnosis_tab()
