"""
归一化层对比工具
Normalization Layer Comparison Tool

对比不同归一化方法的效果
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from utils.visualization.chart_utils import format_number


def normalization_comparison_tab(CHINESE_SUPPORTED):
    """归一化层对比标签页内容"""

    st.header(
        "📏 归一化层对比" if CHINESE_SUPPORTED else "📏 Normalization Layer Comparison"
    )

    st.markdown(
        """
    对比三种主要归一化方法的效果：
    - **BatchNorm**: 在批次维度归一化
    - **LayerNorm**: 在特征维度归一化  
    - **GroupNorm**: 在组内归一化
    """
    )

    # 参数控制
    col1, col2 = st.columns(2)

    with col1:
        num_channels = st.slider(
            "通道数" if CHINESE_SUPPORTED else "Number of Channels",
            min_value=4,
            max_value=128,
            value=16,
            step=4,
            key="norm_channels",
        )

    with col2:
        batch_size = st.slider(
            "批次大小" if CHINESE_SUPPORTED else "Batch Size",
            min_value=4,
            max_value=64,
            value=32,
            step=4,
            key="norm_batch_size",
        )

    # 简化的归一化对比
    if st.button("🚀 生成数据并对比", type="primary"):
        with st.spinner("生成数据并计算..."):
            # 创建测试数据
            torch.manual_seed(42)
            spatial_size = 16  # 保持较小值以确保性能

            x = (
                torch.randn(batch_size, num_channels, spatial_size, spatial_size) * 10
                + 5
            )

            # 显示原始数据统计
            st.info(
                f"""
            **原始数据统计**：
            - 形状: [{batch_size}, {num_channels}, {spatial_size}, {spatial_size}]
            - 均值: {x.mean():.4f}
            - 标准差: {x.std():.4f}
            - 范围: [{x.min():.4f}, {x.max():.4f}]
            """
            )

            # 简化的归一化计算
            # BatchNorm
            batch_norm_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_norm_std = x.std(dim=[0, 2, 3], keepdim=True)
            batch_norm = (x - batch_norm_mean) / (batch_norm_std + 1e-5)

            # LayerNorm
            layer_norm_mean = x.mean(dim=-1, keepdim=True)
            layer_norm_std = x.std(dim=-1, keepdim=True)
            layer_norm = (x - layer_norm_mean) / (layer_norm_std + 1e-5)

            # 显示关键指标
            st.markdown("#### 📈 归一化后的统计量")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("原始均值", f"{x.mean():.4f}")
                st.metric("原始标准差", f"{x.std():.4f}")

            with col2:
                st.metric("BatchNorm均值", f"{batch_norm.mean():.6f}")
                st.metric("BatchNorm标准差", f"{batch_norm.std():.6f}")

            with col3:
                st.metric("LayerNorm均值", f"{layer_norm.mean():.6f}")
                st.metric("LayerNorm标准差", f"{layer_norm.std():.6f}")

            with col4:
                st.metric("数据形状", f"{x.shape}")

            st.success("✅ 观察：归一化后，均值≈0、标准差≈1")

            # 简化的可视化
            st.markdown("#### 📊 激活值分布")

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=x.flatten().numpy(), name="原始数据", opacity=0.7, nbinsx=50
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=batch_norm.flatten().numpy(),
                    name="BatchNorm",
                    opacity=0.7,
                    nbinsx=50,
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=layer_norm.flatten().numpy(),
                    name="LayerNorm",
                    opacity=0.7,
                    nbinsx=50,
                )
            )

            fig.update_layout(
                title="激活值分布对比",
                xaxis_title="激活值",
                yaxis_title="频次",
                barmode="overlay",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        **关键观察**：
        - **BatchNorm**: 在batch维度归一化，适合CNN
        - **LayerNorm**: 在特征维度归一化，适合Transformer
        - 所有方法都将数据调整到均值≈0、标准差≈1
        """
    )

    # ==================== 适用场景分析 ====================
    st.markdown("---")
    st.markdown("### 🎯 适用场景分析与问题诊断")

    st.info('💡 根据项目定位：不仅展示"能用"，更要说明"什么时候会出问题"')

    # 自动检测和建议
    st.markdown("#### 🔍 自动场景检测")

    # 检测batch size
    if batch_size < 8:
        st.error(f"⚠️ **Batch Size过小**: 当前={batch_size}")
        st.write("**问题**: BatchNorm在小batch时统计量不准确，导致训练不稳定")
        st.write("**建议**: 使用GroupNorm或LayerNorm")
    elif batch_size < 16:
        st.warning(f"⚠️ **Batch Size较小**: 当前={batch_size}")
        st.write("**问题**: BatchNorm的效果可能不够稳定")
        st.write("**建议**: 增加batch size或考虑GroupNorm")
    else:
        st.success(f"✅ **Batch Size合适**: 当前={batch_size}，BatchNorm可以正常工作")

    st.markdown("---")

    # 详细对比表格（使用markdown）
    st.markdown("#### 📊 归一化方法详细对比")

    comparison_table = """
| 特性 | BatchNorm | LayerNorm | GroupNorm |
|:-----|:----------|:----------|:----------|
| **归一化维度** | Batch维度 [N, H, W] | 特征维度 [C, H, W] | 组内维度 [G, C/G, H, W] |
| **依赖Batch** | ✅ 强依赖 | ❌ 不依赖 | ❌ 不依赖 |
| **最小Batch** | ≥16 | 1 | 1 |
| **训练/推理一致** | ❌ 不一致 | ✅ 一致 | ✅ 一致 |
| **适用架构** | CNN | Transformer, RNN | CNN (小batch) |
| **计算开销** | 低 | 低 | 中 |
| **参数量** | 2C | 2C | 2C |
| **典型应用** | ResNet, VGG | BERT, GPT | YOLO, Mask R-CNN |
"""

    st.markdown(comparison_table)

    st.markdown("---")

    # 适用场景详细分析
    st.markdown("#### 🎯 何时使用哪种归一化？")

    tab1, tab2, tab3 = st.tabs(["BatchNorm", "LayerNorm", "GroupNorm"])

    with tab1:
        st.markdown("### BatchNorm (Batch Normalization)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ 适用场景**")
            st.markdown(
                """
- **CNN图像任务** (ResNet, VGG, EfficientNet)
- **Batch size ≥ 16** (越大越稳定)
- **训练数据分布一致** (训练=推理)
- **需要最快速度** (计算最高效)
            """
            )

            st.success(
                f"""
**当前配置适合BatchNorm**: {"✅ 是" if batch_size >= 16 else "❌ 否"}
- Batch size = {batch_size}
- 通道数 = {num_channels}
            """
            )

        with col2:
            st.markdown("**❌ 不适用场景**")
            st.markdown(
                """
- **小batch训练** (batch < 8)
- **序列长度变化** (NLP任务)
- **RNN/LSTM** (时序任务)
- **推理单张图片** (统计量不准)
- **在线学习** (数据分布变化)
            """
            )

            if batch_size < 8:
                st.error(
                    """
**❌ 当前不适合BatchNorm**
- Batch太小会导致统计量噪声大
- 建议切换到GroupNorm或LayerNorm
                """
                )

        st.markdown("---")
        st.markdown("**🔧 常见问题与解决方案**")

        problems_table = """
| 问题 | 症状 | 原因 | 解决方案 |
|:-----|:-----|:-----|:---------|
| 训练不稳定 | Loss震荡、不收敛 | Batch太小(<8) | 增大batch或用GroupNorm |
| 推理效果差 | 训练好但推理差 | 训练/推理分布不一致 | 使用moving average或LayerNorm |
| 梯度爆炸 | Loss变NaN | 初始化不当 | 使用He/Xavier初始化 |
| 速度慢 | 训练时间长 | Batch太大 | 减小batch或用混合精度 |
"""
        st.markdown(problems_table)

        st.markdown("**📚 PyTorch实现对照**")
        st.code(
            """
# PyTorch中的BatchNorm
import torch.nn as nn

# 对于4D输入 (N, C, H, W)
bn = nn.BatchNorm2d(num_features=num_channels)

# 关键参数
# - momentum: 移动平均的动量 (默认0.1)
# - eps: 防止除零 (默认1e-5)
# - track_running_stats: 是否追踪统计量 (默认True)

output = bn(input)
        """,
            language="python",
        )

    with tab2:
        st.markdown("### LayerNorm (Layer Normalization)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ 适用场景**")
            st.markdown(
                """
- **Transformer** (BERT, GPT, ViT)
- **RNN/LSTM** (语言模型)
- **任何batch size** (包括1)
- **序列长度变化** (NLP任务)
- **在线学习** (单样本更新)
            """
            )

            st.success(
                """
**LayerNorm总是适用**
- 不依赖batch size
- 训练=推理
- 对序列友好
            """
            )

        with col2:
            st.markdown("**❌ 不适用场景**")
            st.markdown(
                """
- **需要batch统计的场景** (罕见)
- **极度追求速度的CNN** (BatchNorm更快)
            """
            )

            st.info(
                """
**💡 为什么Transformer用LayerNorm？**
- 序列长度变化 → BatchNorm不适用
- Attention机制 → 需要稳定的归一化
- 自回归生成 → batch=1，BatchNorm失效
            """
            )

        st.markdown("---")
        st.markdown("**🔧 常见问题与解决方案**")

        problems_table = """
| 问题 | 症状 | 原因 | 解决方案 |
|:-----|:-----|:-----|:---------|
| 速度比BatchNorm慢 | 训练时间长 | 需要更多计算 | 可接受，或用混合精度 |
| 某些CNN效果不如BatchNorm | 准确率低1-2% | CNN更适合BatchNorm | 权衡速度vs效果 |
| 梯度消失 | 深层网络不收敛 | LayerNorm位置不当 | 调整LayerNorm位置 |
"""
        st.markdown(problems_table)

        st.markdown("**📚 PyTorch实现对照**")
        st.code(
            """
# PyTorch中的LayerNorm
import torch.nn as nn

# 对于任意形状的输入
ln = nn.LayerNorm(normalized_shape=[num_channels, spatial_size, spatial_size])

# 或者只归一化最后几个维度
ln = nn.LayerNorm(normalized_shape=num_channels)

# 关键参数
# - normalized_shape: 要归一化的形状
# - eps: 防止除零 (默认1e-5)
# - elementwise_affine: 是否学习缩放和平移 (默认True)

output = ln(input)
        """,
            language="python",
        )

    with tab3:
        st.markdown("### GroupNorm (Group Normalization)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ 适用场景**")
            st.markdown(
                """
- **小batch CNN** (目标检测、分割)
- **Batch size < 8** 的任何任务
- **YOLO、Mask R-CNN** 等模型
- **折中方案** (性能接近BatchNorm但不依赖batch)
            """
            )

            is_suitable = batch_size < 16
            st.success(
                f"""
**当前配置{"适合" if is_suitable else "可选"}GroupNorm**: {"✅" if is_suitable else "⚠️"}
- 小batch时的最佳选择
- 性能接近BatchNorm
- 不依赖batch统计
            """
            )

        with col2:
            st.markdown("**❌ 不适用场景**")
            st.markdown(
                """
- **大batch CNN** (batch≥32，用BatchNorm更好)
- **Transformer** (直接用LayerNorm)
- **RNN/LSTM** (用LayerNorm)
            """
            )

            if batch_size >= 32:
                st.info(
                    """
**💡 Batch足够大时**
- BatchNorm通常效果更好
- GroupNorm是BatchNorm的近似
- 但GroupNorm更稳定
                """
                )

        st.markdown("---")
        st.markdown("**🔧 常见问题与解决方案**")

        problems_table = """
| 问题 | 症状 | 原因 | 解决方案 |
|:-----|:-----|:-----|:---------|
| 不知道设置多少组 | 效果不稳定 | 组数不合适 | 通常用32组，或C//8 |
| 比BatchNorm慢 | 训练时间长 | 计算开销大 | 可接受，小batch必须 |
| 通道数不能整除 | 报错 | 通道数 % 组数 ≠ 0 | 调整组数使其整除 |
"""
        st.markdown(problems_table)

        st.markdown("**📚 PyTorch实现对照**")
        st.code(
            f"""
# PyTorch中的GroupNorm
import torch.nn as nn

# num_groups: 分组数量（通常用32或通道数//8）
# num_channels: 通道数
gn = nn.GroupNorm(num_groups=32, num_channels={num_channels})

# 组数的选择
# - 32: 经典选择，适合大部分情况
# - num_channels // 8: 自适应选择
# - num_channels: 等价于LayerNorm
# - 1: 等价于LayerNorm (特殊情况)

# 关键参数
# - num_groups: 分组数量
# - num_channels: 通道数 (必须能被num_groups整除)
# - eps: 防止除零 (默认1e-5)

output = gn(input)
        """,
            language="python",
        )

    st.markdown("---")

    # 决策树
    st.markdown("#### 🌳 归一化方法选择决策树")

    st.markdown(
        """
```
开始
  │
  ├─ 是CNN任务？
  │   ├─ 是 → Batch size ≥ 16？
  │   │        ├─ 是 → ✅ 使用 BatchNorm
  │   │        └─ 否 → ✅ 使用 GroupNorm (32组)
  │   │
  │   └─ 否 → 是Transformer/RNN？
  │            ├─ 是 → ✅ 使用 LayerNorm
  │            └─ 否 → 是小batch？
  │                     ├─ 是 → ✅ 使用 GroupNorm 或 LayerNorm
  │                     └─ 否 → ✅ 使用 BatchNorm
```
    """
    )

    # 性能对比
    st.markdown("---")
    st.markdown("#### ⚡ 性能与效果对比")

    performance_table = """
| 指标 | BatchNorm | LayerNorm | GroupNorm |
|:-----|:----------|:----------|:----------|
| **训练速度** | 🟢 最快 | 🟡 中等 | 🟡 中等 |
| **内存占用** | 🟢 最低 | 🟢 最低 | 🟢 最低 |
| **CNN效果** | 🟢 最好 | 🟡 稍差1-2% | 🟢 接近BatchNorm |
| **Transformer效果** | 🔴 不适用 | 🟢 标准选择 | 🟡 可用 |
| **小batch稳定性** | 🔴 差 | 🟢 优秀 | 🟢 优秀 |
| **推理一致性** | 🟡 需moving avg | 🟢 完全一致 | 🟢 完全一致 |
| **实现复杂度** | 🟡 中等 | 🟢 简单 | 🟡 中等 |
"""

    st.markdown(performance_table)

    st.markdown("---")

    # 实战建议
    st.markdown("#### 💡 实战建议")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 推荐组合**")
        st.markdown(
            """
1. **ResNet/VGG (图像分类)**
   - BatchNorm + ReLU
   - Batch size ≥ 32

2. **BERT/GPT (NLP)**
   - LayerNorm + GELU
   - 任何batch size

3. **YOLO/Faster R-CNN (目标检测)**
   - GroupNorm (32组) + ReLU
   - Batch size 通常 < 8

4. **ViT (Vision Transformer)**
   - LayerNorm + GELU
   - 任何batch size

5. **小batch实验**
   - GroupNorm 或 LayerNorm
   - Batch size < 16
        """
        )

    with col2:
        st.markdown("**⚠️ 常见错误**")
        st.markdown(
            """
1. ❌ **小batch用BatchNorm**
   - Batch < 8时BatchNorm非常不稳定
   - 切换到GroupNorm

2. ❌ **Transformer用BatchNorm**
   - 序列长度变化导致失效
   - 必须用LayerNorm

3. ❌ **推理时忘记eval模式**
   - BatchNorm训练和推理不同
   - 记得调用model.eval()

4. ❌ **GroupNorm组数设置不当**
   - 太多组 → 接近LayerNorm
   - 太少组 → 效果差
   - 建议：32组或通道数//8

5. ❌ **混用不同归一化**
   - 同一网络内保持一致
   - 除非有特殊设计
        """
        )

    st.markdown("---")

    # 总结
    st.success(
        """
    ✅ **关键要点总结**：
    
    1. **BatchNorm**: CNN的标准选择，但需要大batch (≥16)
    2. **LayerNorm**: Transformer/RNN的标准选择，不依赖batch
    3. **GroupNorm**: 小batch CNN的救星，性能接近BatchNorm
    
    4. **什么时候会出问题？**
       - BatchNorm + 小batch → 训练不稳定
       - BatchNorm + Transformer → 完全不适用
       - LayerNorm + 大batch CNN → 效果稍差但可用
       - GroupNorm + 通道数不能整除 → 报错
    
    5. **如何选择？**
       - 看任务类型（CNN vs Transformer）
       - 看batch size（大 vs 小）
       - 看训练稳定性要求
       - 看推理场景（单张 vs 批量）
    """
    )
