"""
Vision Transformer (ViT) 分析
Vision Transformer Analysis

展示ViT的核心机制:Patch Embedding|Self-Attention|Position Encoding
核心理念:用可视化展示"如何把图像变成序列"
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image

from utils.vit_models import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    VisionTransformer,
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    get_vit_info,
)
from utils.input_config import get_preset_shape
from utils.example_generator import get_dynamic_example


def visualize_patch_embedding(img_size=224, patch_size=16):
    """
    可视化Patch Embedding过程

    Args:
        img_size: 图像尺寸
        patch_size: Patch大小

    Returns:
        fig: Plotly图表
    """
    num_patches_per_side = img_size // patch_size
    num_patches = num_patches_per_side**2

    # 创建模拟图像(网格)
    img = np.zeros((img_size, img_size, 3))

    # 绘制网格线
    for i in range(0, img_size, patch_size):
        img[i : i + 2, :] = [1, 0, 0]  # 红色水平线
        img[:, i : i + 2] = [0, 0, 1]  # 蓝色垂直线

    # 给每个patch标号
    patch_labels = []
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            patch_idx = i * num_patches_per_side + j
            patch_labels.append(
                {
                    "x": j * patch_size + patch_size // 2,
                    "y": i * patch_size + patch_size // 2,
                    "text": str(patch_idx),
                }
            )

    # 创建图表
    fig = go.Figure()

    # 显示图像
    fig.add_trace(go.Image(z=img))

    # 添加patch编号
    for label in patch_labels:
        fig.add_annotation(
            x=label["x"],
            y=label["y"],
            text=label["text"],
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=2,
        )

    fig.update_layout(
        title=f"Patch Embedding: {img_size}x{img_size} 图像切分成 {num_patches} 个 {patch_size}x{patch_size} patches",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=500,
        width=500,
    )

    return fig


def visualize_attention_weights(attn_weights, img_size=224, patch_size=16):
    """
    可视化Attention权重

    Args:
        attn_weights: 注意力权重 [num_heads, num_patches+1, num_patches+1]
        img_size: 图像尺寸
        patch_size: Patch大小

    Returns:
        fig: Plotly图表
    """
    num_heads = attn_weights.shape[0]

    # 只显示前4个头
    heads_to_show = min(4, num_heads)

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[f"Head {i}" for i in range(heads_to_show)]
    )

    for idx in range(heads_to_show):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # 取第一个token(CLS)对所有token的注意力
        attn = attn_weights[idx, 0, 1:].detach().cpu().numpy()  # 去掉CLS token

        # reshape成2D
        num_patches_per_side = img_size // patch_size
        attn_2d = attn.reshape(num_patches_per_side, num_patches_per_side)

        # 添加热力图
        fig.add_trace(
            go.Heatmap(z=attn_2d, colorscale="Viridis", showscale=(idx == 0)),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Self-Attention权重可视化(CLS Token对各Patch的注意力)", height=600
    )

    return fig


def compare_vit_cnn_params():
    """
    对比ViT和CNN的参数量

    Returns:
        fig: Plotly图表
    """
    models_info = {
        "ResNet-50": 25.6,
        "ResNet-101": 44.5,
        "ViT-Tiny": 5.7,
        "ViT-Small": 22.0,
        "ViT-Base": 86.0,
        "ViT-Large": 307.0,
    }

    fig = go.Figure()

    colors = ["red" if "ResNet" in name else "blue" for name in models_info.keys()]

    fig.add_trace(
        go.Bar(
            x=list(models_info.keys()),
            y=list(models_info.values()),
            text=[f"{v:.1f}M" for v in models_info.values()],
            textposition="auto",
            marker_color=colors,
        )
    )

    fig.update_layout(
        title="ViT vs CNN 参数量对比",
        xaxis_title="模型",
        yaxis_title="参数量 (Million)",
        height=400,
    )

    return fig


def explain_vit_architecture():
    """展示ViT的架构原理"""
    st.markdown(
        """
    ### 🏗️ Vision Transformer (ViT) 架构原理
    
    #### 核心思想:把图像当作序列处理
    """
    )

    st.markdown(
        """
**传统CNN**:
```
图像 -> 卷积层 -> 池化层 -> ... -> 全连接层 -> 分类
```

**ViT**:
```
图像 -> Patch Embedding -> Transformer Encoder -> 分类
```

#### ViT的四大核心组件

**1. Patch Embedding(图像切片)**
    """
    )

    try:
        example = get_dynamic_example("vit")
        calc = example["calculation"]

        st.markdown(
            f"""
        输入图像: [B, 3, {example['img_size']}, {example['img_size']}]  
        {calc['patches']}  
        {calc['embedding']}  
        输出: [B, {example['num_patches']}, {example['d_model']}]
        """
        )

        st.markdown(
            f"""
        **实现方式**: 使用Conv2d(3, {example['d_model']}, kernel_size={example['patch_size']}, stride={example['patch_size']})
        - 等价于将每个{example['patch_size']}x{example['patch_size']}的patch线性投影到{example['d_model']}维
        """
        )
    except Exception as e:
        # 如果动态生成失败,使用默认示例
        st.markdown(
            """
        输入图像: [B, 3, 224, 224]
        切分patches: 224/16 = 14, 共14x14=196个patches
        每个patch: [3, 16, 16] = 768维向量
        输出: [B, 196, 768]
        """
        )

        st.markdown(
            """
        **实现方式**: 使用Conv2d(3, 768, kernel_size=16, stride=16)
        - 等价于将每个16x16的patch线性投影到768维
        """
        )

    st.markdown(
        """
    **2. Position Embedding(位置编码)**
    ```python
    为什么需要?Transformer没有位置信息!
    
    可学习位置编码: [1, 197, 768]  # 196个patches + 1个CLS token
    添加方式: x = x + pos_embed
    ```
    
    **3. [CLS] Token(分类标记)**
    ```python
    作用: 用于分类的特殊token
    初始化: [1, 1, 768] 可学习参数
    位置: 插入到序列开头
    
    [CLS, patch_1, patch_2, ..., patch_196]
    ```
    
    **4. Self-Attention(自注意力)**
    ```python
    Q = X @ W_q
    K = X @ W_k  
    V = X @ W_v
    Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    复杂度: O(N^2),N=196(patches数量)
    ```
    
    #### ViT vs CNN 的关键差异
    
    | 特性 | CNN | ViT |
    |------|-----|-----|
    | **归纳偏置** | 强(平移不变性|局部性) | 弱(需要从数据学习) |
    | **感受野** | 逐层增长 | 全局(第一层就能看到整个图像) |
    | **数据需求** | 小数据集也能work | 需要大数据集预训练 |
    | **计算复杂度** | O(N) | O(N^2) |
    | **参数量** | 相对较少 | 相对较多 |
    
    #### 为什么ViT需要更多数据?
    
    **CNN的优势**:
    - 卷积操作内置了平移不变性(translation invariance)
    - 局部连接天然适合图像的空间结构
    - 可以用少量数据学到好的特征
    
    **ViT的劣势**:
    - 没有内置归纳偏置,需要从数据中学习
    - 在小数据集上容易过拟合
    - 需要大规模预训练 (ImageNet 21K, JFT 300M)
    
    **实验数据**:
    - 小数据集(ImageNet 1K): CNN > ViT
    - 大数据集(ImageNet 21K): ViT ≈ CNN
    - 超大数据集(JFT 300M): ViT > CNN
    """
    )


def vit_analysis_tab(chinese_supported=True):
    """Vision Transformer分析主函数"""

    st.header("🔍 Vision Transformer (ViT) 分析")
    st.markdown(
        """
    > **核心问题**:Transformer如何应用到图像领域?ViT和CNN有什么本质区别?
    
    **验证方法**:可视化Patch Embedding|Self-Attention,对比ViT和CNN
    """
    )

    st.markdown("---")

    # 架构原理
    with st.expander("🏗️ ViT架构原理(点击展开)", expanded=False):
        explain_vit_architecture()

    st.markdown("---")

    # Patch Embedding可视化
    st.subheader("📐 1. Patch Embedding可视化")

    col1, col2 = st.columns(2)

    with col1:
        img_size = st.selectbox("图像尺寸", [224, 384], index=0)

    with col2:
        patch_size = st.selectbox("Patch大小", [16, 32], index=0)

    num_patches = (img_size // patch_size) ** 2

    st.info(
        f"""
    **计算过程**:
    - 图像尺寸: {img_size}x{img_size}
    - Patch大小: {patch_size}x{patch_size}
    - Patches数量: ({img_size}/{patch_size})^2 = **{num_patches}个**
    - 每个patch维度: {patch_size}x{patch_size}x3 = {patch_size*patch_size*3}
    """
    )

    # 显示切分可视化
    fig1 = visualize_patch_embedding(img_size, patch_size)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        """
    **关键理解**:
    - 图像被切分成不重叠的patches
    - 每个patch通过线性投影变成embedding
    - 实现方式:Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    """
    )

    # ViT模型对比
    st.markdown("---")
    st.subheader("⚖️ 2. ViT模型规模对比")

    model_choice = st.selectbox(
        "选择ViT模型",
        ["vit_tiny", "vit_small", "vit_base"],
        format_func=lambda x: {
            "vit_tiny": "ViT-Tiny (5.7M参数)",
            "vit_small": "ViT-Small (22M参数)",
            "vit_base": "ViT-Base (86M参数)",
        }[x],
    )

    # 显示模型信息
    vit_info = get_vit_info(model_choice)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**模型配置**")
        config = vit_info["config"]
        st.code(
            f"""
Embedding维度: {config['embed_dim']}
Transformer层数: {config['depth']}
注意力头数: {config['num_heads']}
Patch大小: {config['patch_size']}x{config['patch_size']}
预估参数量: {config['params_estimate']}
        """
        )

    with col2:
        st.markdown("**架构细节**")
        arch = vit_info["architecture"]
        st.code(
            f"""
Patch Embedding:
  {arch['patch_embedding']}

Position Embedding:
  {arch['position_embedding']}

Transformer Blocks:
  {arch['transformer_blocks']}
  {arch['attention']}
        """
        )

    # 参数量对比图
    st.markdown("#### 📊 ViT vs CNN 参数量对比")
    fig2 = compare_vit_cnn_params()
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        """
    **观察**:
    - ViT-Base (86M) 参数量约为 ResNet-50 (25M) 的3.4倍
    - ViT-Large (307M) 参数量非常大,需要大规模数据预训练
    - ViT-Tiny (5.7M) 适合资源受限场景
    """
    )

    # Self-Attention可视化
    st.markdown("---")
    st.subheader("👁️ 3. Self-Attention可视化")

    st.markdown(
        """
    **Self-Attention的作用**:
    - 每个patch可以"看到"所有其他patches
    - 第一层就有全局感受野(与CNN不同)
    - 注意力权重反映了patches之间的关系
    """
    )

    if st.button("🚀 生成随机数据并可视化Attention", type="primary"):
        with st.spinner("计算中..."):
            # 创建模型
            if model_choice == "vit_tiny":
                model = create_vit_tiny(img_size=224, num_classes=10)
            elif model_choice == "vit_small":
                model = create_vit_small(img_size=224, num_classes=10)
            else:
                model = create_vit_base(img_size=224, num_classes=10)

            model.eval()

            # 生成随机输入
            x = torch.randn(1, 3, 224, 224)

            # 前向传播并获取attention weights
            with torch.no_grad():
                _, attn_weights_list = model(x, return_attention=True)

            # 显示第一层的attention
            first_layer_attn = attn_weights_list[0][0]  # [num_heads, 197, 197]

            st.success(f"✅ 成功获取Attention权重,形状: {first_layer_attn.shape}")

            # 可视化
            fig3 = visualize_attention_weights(
                first_layer_attn, img_size=224, patch_size=16
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown(
                """
            **解读**:
            - 热力图显示了[CLS] token对各个patch的注意力分布
            - 不同的attention head关注不同的区域
            - 亮色区域表示高注意力,暗色区域表示低注意力
            """
            )

    # 计算复杂度分析
    st.markdown("---")
    st.subheader("⚡ 4. 计算复杂度分析")

    st.markdown(
        """
    ### Self-Attention的计算复杂度
    
    对于输入序列长度N(patches数量):
    
    **时间复杂度**:
    ```
    Q @ K^T: O(N^2 · d)    # Nxd 矩阵乘以 dxN 矩阵
    Softmax: O(N^2)        # 对NxN矩阵做softmax
    Attn @ V: O(N^2 · d)   # NxN 矩阵乘以 Nxd 矩阵
    
    总复杂度: O(N^2 · d)
    ```
    
    **空间复杂度**:
    ```
    存储attention矩阵: O(N^2)
    ```
    
    ### 与CNN对比
    
    | 操作 | 复杂度 | 说明 |
    |------|--------|------|
    | **Self-Attention** | O(N^2·d) | N=patches数量,随图像尺寸平方增长 |
    | **卷积** | O(k^2·d^2·N) | k=kernel大小,d=通道数,N=特征图大小 |
    
    ### 实际数值
    
    假设224x224图像,patch_size=16:
    - N = (224/16)^2 = 196个patches
    - Attention矩阵: 196x196 = 38,416个元素
    - 12个头: 12x38,416 = 460,992个元素
    
    **结论**:
    - ViT的计算量随图像尺寸平方增长
    - 高分辨率图像(如1024x1024)计算量巨大
    - 需要用到各种优化技巧(如Linformer|Performer等)
    """
    )

    # 适用场景
    st.markdown("---")
    st.subheader("🎯 5. ViT vs CNN:何时使用?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ 使用ViT的场景")
        st.markdown(
            """
        1. **有大规模预训练模型**
           - 使用ImageNet-21k预训练权重
           - 或者JFT-300M等超大数据集
        
        2. **目标任务数据量充足**
           - 至少有几万张标注图像
           - 或者可以做数据增强
        
        3. **需要全局建模能力**
           - 目标检测|实例分割
           - 需要长距离依赖关系
        
        4. **计算资源充足**
           - 有GPU/TPU支持
           - 可以接受较长的训练时间
        """
        )

    with col2:
        st.markdown("### ✅ 使用CNN的场景")
        st.markdown(
            """
        1. **数据量较小**
           - 只有几千张图像
           - 难以获取大规模数据
        
        2. **需要快速训练**
           - 资源受限
           - 需要边缘部署
        
        3. **任务依赖局部特征**
           - 纹理分类
           - 边缘检测
        
        4. **需要平移不变性**
           - 目标位置不固定
           - 需要泛化到不同位置
        """
        )

    # 总结
    st.markdown("---")
    st.subheader("💡 核心要点")

    st.markdown(
        """
    ### ViT的革命性贡献
    
    1. **证明了Transformer可以应用到视觉领域**
       - 打破了CNN在视觉任务上的垄断
       - 开启了视觉Transformer的研究热潮
    
    2. **展示了scaling law的威力**
       - 模型越大+数据越多 = 性能越好
       - ViT-Huge在JFT-300M上达到了SOTA
    
    3. **简化了模型设计**
       - 不需要精心设计的卷积结构
       - 统一的Transformer架构
    
    ### 实际工程建议
    
    **如果你是...**
    
    - **学生/研究者**: 使用预训练的ViT(timm库),在自己的数据上微调
    - **工业界**: 小数据集用CNN,大数据集用ViT
    - **边缘设备**: 优先考虑MobileNet|EfficientNet等轻量CNN
    - **云端部署**: 可以使用ViT-Base或ViT-Large
    
    ### 记住三个关键数字
    
    - **196**: 224x224图像使用16x16 patch得到的序列长度
    - **768**: ViT-Base的embedding维度
    - **12**: ViT-Base的Transformer层数和注意力头数
    """
    )


if __name__ == "__main__":
    # 测试运行
    vit_analysis_tab()
