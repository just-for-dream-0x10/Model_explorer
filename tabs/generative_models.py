"""
生成模型数学原理模块

包含：Autoencoder、VAE、GAN、Diffusion Model 的数学原理与交互式演示。
所有计算使用 numpy，不依赖 torch。
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.visualization.chart_utils import ChartBuilder
from simple_latex import display_latex


# ============================================================
# 缓存计算函数
# ============================================================

@st.cache_data
def generate_ae_latent_clusters(n_classes=10, points_per_class=50, seed=42):
    """生成模拟的 Autoencoder 潜在空间聚类数据"""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, 2) * 2.0
    labels = []
    z1_list, z2_list = [], []
    for cls in range(n_classes):
        z = centers[cls] + rng.randn(points_per_class, 2) * 0.5
        z1_list.append(z[:, 0])
        z2_list.append(z[:, 1])
        labels.extend([cls] * points_per_class)
    return np.concatenate(z1_list), np.concatenate(z2_list), np.array(labels)


@st.cache_data
def generate_vae_latent_gaussians(n_classes=10, points_per_class=50, seed=42):
    """生成模拟的 VAE 潜在空间高斯分布数据（含不确定性）"""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, 2) * 2.0
    all_z1, all_z2, all_labels = [], [], []
    all_sigma = []
    for cls in range(n_classes):
        mu = centers[cls]
        sigma = rng.uniform(0.3, 0.8)
        z = mu + rng.randn(points_per_class, 2) * sigma
        all_z1.append(z[:, 0])
        all_z2.append(z[:, 1])
        all_labels.extend([cls] * points_per_class)
        all_sigma.extend([sigma] * points_per_class)
    return (np.concatenate(all_z1), np.concatenate(all_z2),
            np.array(all_labels), np.array(all_sigma))


@st.cache_data
def simulate_gan_training(n_steps, lr_g, lr_d, seed=42):
    """模拟 GAN 训练过程中的 loss 曲线"""
    rng = np.random.RandomState(seed)
    steps = np.arange(1, n_steps + 1)

    # D loss: 从高开始下降，最终趋近于 -log(2) ≈ -0.693
    d_loss_init = 2.0
    d_loss_final = -0.693
    d_decay = 1.0 - np.exp(-lr_d * steps / 50.0)
    d_loss = d_loss_init + (d_loss_final - d_loss_init) * d_decay
    d_loss += rng.randn(n_steps) * 0.08 * np.exp(-steps / (n_steps * 0.3))

    # G loss: 从高开始下降，最终趋近于 -log(2) ≈ -0.693
    g_loss_init = 3.0
    g_loss_final = -0.693
    g_decay = 1.0 - np.exp(-lr_g * steps / 60.0)
    g_loss = g_loss_init + (g_loss_final - g_loss_init) * g_decay
    g_loss += rng.randn(n_steps) * 0.1 * np.exp(-steps / (n_steps * 0.25))

    return steps.tolist(), d_loss.tolist(), g_loss.tolist()


@st.cache_data
def compute_noise_schedule(schedule_type, T, beta_start=1e-4, beta_end=0.02):
    """计算噪声调度 schedule"""
    if schedule_type == "线性 (Linear)":
        betas = np.linspace(beta_start, beta_end, T)
    else:  # cosine
        s = 0.008
        steps = np.arange(T + 1)
        f_t = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod = alphas_cumprod[:-1]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, beta_start, beta_end)
        if len(betas) < T:
            betas = np.concatenate([betas, np.full(T - len(betas), beta_end)])

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    snr = alphas_cumprod / (1.0 - alphas_cumprod)

    return betas, alphas_cumprod, snr


@st.cache_data
def generate_diffusion_images(T, seed=42):
    """生成扩散过程各步骤的模拟图像（同心圆 + 递增噪声）"""
    rng = np.random.RandomState(seed)
    size = 28
    y, x = np.ogrid[:size, :size]
    center = size // 2
    clean = np.sin(np.sqrt((x - center) ** 2 + (y - center) ** 2) * 0.8).astype(float)

    # 选取关键步骤展示
    if T <= 20:
        show_steps = list(range(0, T, max(1, T // 6)))
    else:
        show_steps = [0] + list(np.linspace(0, T - 1, 7, dtype=int)[1:])

    images = {}
    for t in show_steps:
        noise_ratio = t / max(T - 1, 1)
        noise = rng.randn(size, size) * noise_ratio * 1.5
        noisy = clean + noise
        noisy = np.clip(noisy, -2, 2)
        images[t] = noisy

    return images


# ============================================================
# 主函数
# ============================================================

def generative_models_tab(chinese_supported=True):
    """生成模型标签页"""

    st.header("🧬 生成模型 Generative Models 数学原理")

    chart = ChartBuilder()

    # ============================================================
    # Section 1: Autoencoder 自编码器
    # ============================================================
    st.markdown("---")
    st.markdown("## 1. Autoencoder 自编码器")

    with st.expander("💡 核心概念", expanded=True):
        st.markdown("""
        **Autoencoder 是一种无监督学习模型，通过压缩再重建来学习数据的紧凑表示。**

        架构流程：**Encoder（编码器）** → **Latent Space（潜在空间）** → **Decoder（解码器）**

        1. **Encoder**：将高维输入 """)
        display_latex(r"x")
        st.markdown(""" 映射到低维潜在表示 """)
        display_latex(r"z = f(x)")
        st.markdown("""
        2. **Latent Space**：数据的紧凑瓶颈表示
        3. **Decoder**：从潜在表示重建原始数据 """)
        display_latex(r"\hat{x} = g(z)")
        st.markdown("""

        **关键思想**：如果瓶颈层维度远小于输入维度，网络被迫学习数据中最本质的特征。
        """)

    st.markdown("### 1.1 网络架构与参数计算")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**⚙️ 架构配置**")
        input_dim = st.slider("输入维度 input_dim", 64, 1024, 784, step=64,
                              key="ae_input_dim")
        latent_dim = st.slider("潜在维度 latent_dim", 2, 64, 16,
                               key="ae_latent_dim")
        hidden_dims_str = st.text_input("隐藏层维度（逗号分隔）", "256, 128, 64",
                                        key="ae_hidden_dims")

        try:
            hidden_dims = [int(d.strip()) for d in hidden_dims_str.split(",")]
        except ValueError:
            hidden_dims = [256, 128, 64]
            st.warning("隐藏层维度格式错误，已使用默认值 [256, 128, 64]")

        # 构建完整架构
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        st.markdown("**📐 架构示意图**")
        st.markdown("```")
        arch_str = "Encoder:  "
        for i, d in enumerate(encoder_dims):
            if i < len(encoder_dims) - 1:
                arch_str += f"[{d}] → "
            else:
                arch_str += f"[{d}]"
        st.code(arch_str)
        arch_str2 = "Decoder:  "
        for i, d in enumerate(decoder_dims):
            if i < len(decoder_dims) - 1:
                arch_str2 += f"[{d}] → "
            else:
                arch_str2 += f"[{d}]"
        st.code(arch_str2)
        st.markdown("```")

    with col2:
        st.markdown("**📊 参数量计算**")
        st.markdown("全连接层参数公式：")
        display_latex(r"W \in \mathbb{R}^{n_{out} \times n_{in}}")
        st.markdown("，")
        display_latex(r"b \in \mathbb{R}^{n_{out}}")

        total_params = 0
        param_rows = []
        for i in range(len(encoder_dims) - 1):
            n_in = encoder_dims[i]
            n_out = encoder_dims[i + 1]
            params = n_in * n_out + n_out
            total_params += params
            param_rows.append({
                "层": f"Encoder L{i + 1}",
                "输入": n_in,
                "输出": n_out,
                "权重": n_in * n_out,
                "偏置": n_out,
                "参数量": params
            })

        for i in range(len(decoder_dims) - 1):
            n_in = decoder_dims[i]
            n_out = decoder_dims[i + 1]
            params = n_in * n_out + n_out
            total_params += params
            param_rows.append({
                "层": f"Decoder L{i + 1}",
                "输入": n_in,
                "输出": n_out,
                "权重": n_in * n_out,
                "偏置": n_out,
                "参数量": params
            })

        import pandas as pd
        df_params = pd.DataFrame(param_rows)
        st.dataframe(df_params, use_container_width=True, hide_index=True)
        st.markdown(f"**总参数量**: {total_params:,}")

    st.markdown("### 1.2 损失函数")

    st.markdown("重建损失（MSE）:")
    st.latex(r"L_{recon} = \|x - \hat{x}\|^2 = \sum_{i=1}^{n} (x_i - \hat{x}_i)^2")

    st.markdown("- ")
    display_latex(r"x")
    st.markdown("：原始输入")
    st.markdown("- ")
    display_latex(r"\hat{x}")
    st.markdown("：解码器重建输出")
    st.markdown("- 目标：最小化输入与重建之间的差异")

    st.markdown("### 1.3 潜在空间可视化")

    st.markdown("下图展示模拟的 2D 潜在空间中，不同数字类别的聚类分布：")

    z1, z2, labels = generate_ae_latent_clusters()
    digit_names = [f"数字 {i}" for i in range(10)]

    fig_latent = go.Figure()
    for cls in range(10):
        mask = labels == cls
        fig_latent.add_trace(go.Scatter(
            x=z1[mask], y=z2[mask],
            mode="markers",
            name=digit_names[cls],
            marker=dict(size=5, opacity=0.7),
        ))
    fig_latent.update_layout(
        title="Autoencoder 2D 潜在空间分布",
        xaxis_title="$z_1$", yaxis_title="$z_2$",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    chart.display_chart(fig_latent)

    st.markdown("### 1.4 重建效果演示")

    st.markdown("使用随机数据模拟原始输入与重建输出的对比：")

    rng = np.random.RandomState(42)
    n_samples = 6
    original = rng.randn(n_samples, 28, 28)
    # 模拟重建：添加少量噪声和轻微失真
    reconstructed = original + rng.randn(n_samples, 28, 28) * 0.15
    reconstructed = np.clip(reconstructed, -3, 3)

    fig_recon = make_subplots(
        rows=2, cols=n_samples,
        subplot_titles=[f"样本 {i + 1}" for i in range(n_samples)] * 2,
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
    )

    for i in range(n_samples):
        fig_recon.add_trace(
            go.Heatmap(z=original[i], colorscale="gray", showscale=False),
            row=1, col=i + 1,
        )
        fig_recon.add_trace(
            go.Heatmap(z=reconstructed[i], colorscale="gray", showscale=False),
            row=2, col=i + 1,
        )

    fig_recon.update_layout(
        height=350,
        title=dict(text="原始 vs 重建", x=0.5, font=dict(size=14)),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    # 添加行标注
    fig_recon.add_annotation(
        text="原始", xref="paper", yref="paper", x=-0.02, y=0.75,
        showarrow=False, font=dict(size=13, color="blue"),
    )
    fig_recon.add_annotation(
        text="重建", xref="paper", yref="paper", x=-0.02, y=0.25,
        showarrow=False, font=dict(size=13, color="red"),
    )
    chart.display_chart(fig_recon)

    # ============================================================
    # Section 2: VAE 变分自编码器
    # ============================================================
    st.markdown("---")
    st.markdown("## 2. VAE 变分自编码器")

    with st.expander("💡 核心概念", expanded=True):
        st.markdown("""
        **VAE 在 Autoencoder 基础上引入概率建模，使潜在空间具有连续性和良好的插值性质。**

        与普通 Autoencoder 的关键区别：
        - Encoder 不再输出固定向量，而是输出分布参数 """)
        display_latex(r"\mu")
        st.markdown(""" 和 """)
        display_latex(r"\sigma^2")
        st.markdown("""
        - 潜在表示 """)
        display_latex(r"z")
        st.markdown(""" 从该分布中采样
        - 损失函数增加 KL 散度正则项，使潜在空间接近标准正态分布

        **优势**：可以从潜在空间中随机采样来生成新的、合理的样本。
        """)

    st.markdown("### 2.1 ELBO 目标函数")

    st.markdown("VAE 优化的目标函数 —— Evidence Lower Bound (ELBO)：")
    st.latex(r"\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))")

    st.markdown("**第一项：重建项 (Reconstruction Loss)**")
    st.latex(r"\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{2}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2")

    st.markdown("**第二项：KL 散度正则项**")
    st.latex(r"D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum_{j=1}^{J}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)")

    st.markdown("其中 ")
    display_latex(r"p(z) = \mathcal{N}(0, I)")
    st.markdown(" 是标准正态先验，")
    display_latex(r"q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))")
    st.markdown(" 是编码器输出的后验分布。")

    st.markdown("### 2.2 KL 散度逐步计算")

    col_kl1, col_kl2 = st.columns([1, 1])

    with col_kl1:
        st.markdown("**⚙️ 交互式 KL 散度计算**")
        vae_latent_dim = st.slider("潜在维度 J", 2, 32, 8, key="vae_latent_dim")

        rng_vae = np.random.RandomState(123)
        mu_values = rng_vae.randn(vae_latent_dim) * 0.5
        sigma_values = np.exp(rng_vae.randn(vae_latent_dim) * 0.3)

        st.markdown("**编码器输出的分布参数：**")
        for j in range(min(vae_latent_dim, 8)):
            st.markdown(f"- 维度 j={j}: ")
            display_latex(rf"\mu_{{{j}}} = {mu_values[j]:.4f}")
            st.markdown(", ")
            display_latex(rf"\sigma_{{{j}}} = {sigma_values[j]:.4f}")
            st.markdown(", ")
            display_latex(rf"\sigma_{{{j}}}^2 = {sigma_values[j] ** 2:.4f}")
        if vae_latent_dim > 8:
            st.markdown(f"... (共 {vae_latent_dim} 维)")

    with col_kl2:
        st.markdown("**📊 KL 散度计算过程**")

        kl_per_dim = -0.5 * (1.0 + np.log(sigma_values ** 2) - mu_values ** 2 - sigma_values ** 2)
        total_kl = np.sum(kl_per_dim)

        st.latex(r"KL_j = -\frac{1}{2}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)")

        kl_rows = []
        for j in range(min(vae_latent_dim, 8)):
            kl_rows.append({
                "维度 j": j,
                "mu_j": f"{mu_values[j]:.4f}",
                "sigma_j^2": f"{sigma_values[j] ** 2:.4f}",
                "log(sigma_j^2)": f"{np.log(sigma_values[j] ** 2):.4f}",
                "KL_j": f"{kl_per_dim[j]:.4f}",
            })
        df_kl = pd.DataFrame(kl_rows)
        st.dataframe(df_kl, use_container_width=True, hide_index=True)

        st.markdown("**总 KL 散度**: ")
        display_latex(rf"D_{{KL}} = {total_kl:.4f}")
        st.markdown("**平均每维 KL**: ")
        display_latex(rf"\bar{{KL}} = {total_kl / vae_latent_dim:.4f}")

    st.markdown("### 2.3 重参数化技巧 Reparameterization Trick")

    st.markdown("""
    **问题**：采样操作 """)
    display_latex(r"z \sim q(z|x)")
    st.markdown(""" 不可微，无法反向传播。

    **解决方案**：将随机性从计算图中分离出来：
    """)
    st.latex(r"z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)")

    st.markdown("现在 ")
    display_latex(r"\mu")
    st.markdown(" 和 ")
    display_latex(r"\sigma")
    st.markdown(" 直接参与计算，梯度可以正常回传。")

    st.markdown("### 2.4 采样分布交互演示")

    col_rep1, col_rep2 = st.columns([1, 1])

    with col_rep1:
        st.markdown("**⚙️ 设置分布参数**")
        st.markdown("**")
        display_latex(r"\mu")
        st.markdown("**")
        demo_mu = st.slider("mu", -3.0, 3.0, 0.0, 0.1, key="rep_mu")
        st.markdown("**")
        display_latex(r"\sigma")
        st.markdown("**")
        demo_sigma = st.slider("sigma", 0.1, 3.0, 1.0, 0.1, key="rep_sigma")
        n_samples_demo = st.slider("采样数量", 100, 5000, 2000, key="rep_n_samples")

        # 从 N(mu, sigma^2) 采样
        rng_rep = np.random.RandomState(42)
        epsilon = rng_rep.randn(n_samples_demo)
        z_samples = demo_mu + demo_sigma * epsilon

        # 理论 PDF
        x_range = np.linspace(demo_mu - 4 * demo_sigma, demo_mu + 4 * demo_sigma, 200)
        pdf = (1.0 / (demo_sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_range - demo_mu) / demo_sigma) ** 2
        )

    with col_rep2:
        fig_rep = go.Figure()
        fig_rep.add_trace(go.Histogram(
            x=z_samples, nbinsx=50, histnorm="probability density",
            name="采样分布", marker_color="rgba(100,149,237,0.5)",
            showlegend=True,
        ))
        fig_rep.add_trace(go.Scatter(
            x=x_range, y=pdf, mode="lines",
            name=f"$\\mathcal{{N}}({demo_mu:.1f}, {demo_sigma:.1f}^2)$ 理论PDF",
            line=dict(color="red", width=2),
        ))
        fig_rep.update_layout(
            title=f"采样分布: $z = {demo_mu:.1f} + {demo_sigma:.1f} \\times \\varepsilon$",
            xaxis_title="$z$", yaxis_title="概率密度",
            height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        chart.display_chart(fig_rep)

        st.markdown(f"""
        **采样统计**：
        - 均值: {np.mean(z_samples):.4f}（理论值: {demo_mu:.4f}）
        - 标准差: {np.std(z_samples):.4f}（理论值: {demo_sigma:.4f}）
        """)

    st.markdown("### 2.5 VAE 潜在空间可视化（含不确定性）")

    z1_vae, z2_vae, labels_vae, sigmas_vae = generate_vae_latent_gaussians()

    fig_vae_latent = go.Figure()
    for cls in range(10):
        mask = labels_vae == cls
        avg_sigma = np.mean(sigmas_vae[mask])
        fig_vae_latent.add_trace(go.Scatter(
            x=z1_vae[mask], y=z2_vae[mask],
            mode="markers",
            name=f"数字 {cls} ($\\sigma$={avg_sigma:.2f})",
            marker=dict(
                size=6, opacity=0.6,
                color=ChartBuilder.DEFAULT_COLORS[cls % len(ChartBuilder.DEFAULT_COLORS)],
            ),
        ))
        # 绘制不确定性椭圆（1-sigma）
        cx, cy = np.mean(z1_vae[mask]), np.mean(z2_vae[mask])
        theta = np.linspace(0, 2 * np.pi, 60)
        ex = cx + avg_sigma * np.cos(theta)
        ey = cy + avg_sigma * np.sin(theta)
        fig_vae_latent.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            showlegend=False,
            line=dict(color=ChartBuilder.DEFAULT_COLORS[cls % len(ChartBuilder.DEFAULT_COLORS)],
                      width=1, dash="dash"),
        ))

    fig_vae_latent.update_layout(
        title="VAE 潜在空间分布（虚线为 1-$\\sigma$ 不确定性区域）",
        xaxis_title="$z_1$", yaxis_title="$z_2$",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9)),
    )
    chart.display_chart(fig_vae_latent)

    # ============================================================
    # Section 3: GAN 生成对抗网络
    # ============================================================
    st.markdown("---")
    st.markdown("## 3. GAN 生成对抗网络")

    with st.expander("💡 核心概念", expanded=True):
        st.markdown("""
        **GAN 由两个对抗网络组成：Generator（生成器）和 Discriminator（判别器）。**

        - **Generator** """)
        display_latex(r"G(z)")
        st.markdown("""：从随机噪声 """)
        display_latex(r"z")
        st.markdown(""" 生成伪造样本，目标是骗过判别器
        - **Discriminator** """)
        display_latex(r"D(x)")
        st.markdown("""：判断输入是真实样本还是生成样本，目标是正确区分

        **训练过程**：两个网络进行博弈，最终达到 Nash 均衡 —— 生成器产生逼真的样本，判别器无法区分真假。
        """)

    st.markdown("### 3.1 Minimax 目标函数")

    st.markdown("GAN 的经典 Minimax 目标函数：")
    st.latex(r"\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]")

    st.markdown("**公式解读：**")

    st.markdown("""
    | 符号 | 含义 |
    |------|------|
    | """)
    display_latex(r"D(x)")
    st.markdown(""" | 判别器对真实样本 """)
    display_latex(r"x")
    st.markdown(""" 的输出概率 |
    | """)
    display_latex(r"D(G(z))")
    st.markdown(""" | 判别器对生成样本 """)
    display_latex(r"G(z)")
    st.markdown(""" 的输出概率 |
    | """)
    display_latex(r"\max_D")
    st.markdown(""" | 判别器最大化目标（正确分类） |
    | """)
    display_latex(r"\min_G")
    st.markdown(""" | 生成器最小化目标（骗过判别器） |

    **最优解**：""")
    display_latex(r"D^*(x) = \frac{1}{2}")
    st.markdown("，")
    display_latex(r"p_G = p_{data}")
    st.markdown("（生成分布等于真实分布）。")

    st.markdown("### 3.2 Generator 与 Discriminator 结构")

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("**Generator 生成器**")
        st.markdown("""
        ```
        z (噪声向量) → FC → BN → ReLU → FC → BN → ReLU → FC → Tanh → 生成图像
        ```
        - 输入：""")
        display_latex(r"z \sim \mathcal{N}(0, I)")
        st.markdown("""，通常 100 维
        - 输出：与真实数据同尺寸的生成样本
        - 激活函数：中间层用 ReLU，输出层用 Tanh
        """)

    with col_g2:
        st.markdown("**Discriminator 判别器**")
        st.markdown("""
        ```
        图像输入 → Conv → LeakyReLU → Conv → LeakyReLU → FC → Sigmoid → 真/假概率
        ```
        - 输入：真实图像或生成图像
        - 输出：标量概率 """)
        display_latex(r"D(x) \in [0, 1]")
        st.markdown("""
        - 激活函数：中间层用 LeakyReLU，输出层用 Sigmoid
        """)

    st.markdown("### 3.3 训练过程模拟")

    st.markdown("**⚙️ 调整训练参数，观察 G 和 D 的 loss 变化**")

    col_gan1, col_gan2 = st.columns([1, 2])

    with col_gan1:
        n_steps = st.slider("训练步数", 100, 2000, 500, step=50, key="gan_steps")
        lr_g = st.slider("Generator 学习率", 0.0001, 0.01, 0.002, 0.0001,
                         format="%.4f", key="gan_lr_g")
        lr_d = st.slider("Discriminator 学习率", 0.0001, 0.01, 0.002, 0.0001,
                         format="%.4f", key="gan_lr_d")

        st.markdown("""
        **参数说明：**
        - **G 学习率过高** → 生成器变化太快，判别器跟不上 → 训练不稳定
        - **D 学习率过高** → 判别器过强，生成器梯度消失 → 无法学习
        - **理想情况**：两者学习率相近，交替提升
        """)

    with col_gan2:
        steps, d_loss, g_loss = simulate_gan_training(n_steps, lr_g, lr_d)

        fig_gan = go.Figure()
        fig_gan.add_trace(go.Scatter(
            x=steps, y=d_loss, mode="lines", name="D Loss",
            line=dict(color="#1f77b4", width=2), opacity=0.8,
        ))
        fig_gan.add_trace(go.Scatter(
            x=steps, y=g_loss, mode="lines", name="G Loss",
            line=dict(color="#d62728", width=2), opacity=0.8,
        ))
        # 理论最优线
        fig_gan.add_hline(
            y=-0.693, line_dash="dash", line_color="green",
            annotation_text="理论最优 $-\\log 2 \\approx -0.693$",
            annotation_position="top left",
        )
        fig_gan.update_layout(
            title=f"GAN 训练 Loss 曲线（$\\eta_G={lr_g:.4f}$, $\\eta_D={lr_d:.4f}$）",
            xaxis_title="训练步数", yaxis_title="Loss",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        chart.display_chart(fig_gan)

        # 训练动态分析
        final_d = d_loss[-1]
        final_g = g_loss[-1]
        st.markdown(f"""
        **训练结果分析**：
        - D 最终 Loss: {final_d:.3f}（目标: -0.693）
        - G 最终 Loss: {final_g:.3f}（目标: -0.693）
        - D-G 差距: {abs(final_d - final_g):.3f}
        """)

        if abs(final_d - final_g) < 0.3:
            st.success("G 和 D 趋于均衡，训练收敛良好。")
        elif final_d < final_g - 0.5:
            st.warning("判别器过强，生成器学习困难。建议降低 D 学习率或增加 G 学习率。")
        elif final_g < final_d - 0.5:
            st.warning("生成器过强，判别器被压制。建议降低 G 学习率或增加 D 学习率。")

    # ============================================================
    # Section 4: Diffusion Model 扩散模型
    # ============================================================
    st.markdown("---")
    st.markdown("## 4. Diffusion Model 扩散模型")

    with st.expander("💡 核心概念", expanded=True):
        st.markdown("""
        **扩散模型通过逐步添加噪声（前向过程）和逐步去噪（反向过程）来生成数据。**

        - **前向过程 (Forward Process)**：逐步向数据添加高斯噪声，经过 """)
        display_latex(r"T")
        st.markdown(""" 步后数据变为纯噪声
        - **反向过程 (Reverse Process)**：训练神经网络逐步去除噪声，从纯噪声恢复出数据

        **优势**：训练稳定、生成质量高、理论保证强。
        """)

    st.markdown("### 4.1 前向过程 Forward Process")

    st.markdown("每一步添加少量噪声：")
    st.latex(r"q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \; \beta_t I)")

    st.markdown("可以直接从 ")
    display_latex(r"x_0")
    st.markdown(" 一步到达 ")
    display_latex(r"x_t")
    st.markdown("：")
    st.latex(r"q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, \; (1 - \bar{\alpha}_t) I)")

    st.markdown("其中：")
    st.markdown("- ")
    display_latex(r"\beta_t")
    st.markdown("：第 t 步的噪声调度系数")
    st.markdown("- ")
    display_latex(r"\alpha_t = 1 - \beta_t")
    st.markdown("- ")
    display_latex(r"\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s")
    st.markdown("（累积乘积）")

    st.markdown("### 4.2 反向过程 Reverse Process")

    st.markdown("神经网络学习逐步去噪：")
    st.latex(r"p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \; \mu_\theta(x_t, t), \; \Sigma_\theta(x_t, t))")

    st.markdown("**训练目标**：网络 ")
    display_latex(r"\epsilon_\theta(x_t, t)")
    st.markdown(" 预测添加的噪声 ")
    display_latex(r"\epsilon")
    st.latex(r"L = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]")

    st.markdown("其中 ")
    display_latex(r"x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon")
    st.markdown("。")

    st.markdown("### 4.3 噪声调度 Noise Schedule")

    col_diff1, col_diff2 = st.columns([1, 1])

    with col_diff1:
        st.markdown("**⚙️ 调度参数配置**")
        diff_T = st.slider("扩散步数 T", 10, 1000, 200, step=10, key="diff_T")
        schedule_type = st.selectbox(
            "噪声调度类型",
            ["线性 (Linear)", "余弦 (Cosine)"],
            key="diff_schedule",
        )

        betas, alphas_cumprod, snr = compute_noise_schedule(schedule_type, diff_T)
        steps_arr = np.arange(1, diff_T + 1)

    with col_diff2:
        # Beta schedule
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=steps_arr, y=betas, mode="lines",
            name="$\\beta_t$", line=dict(color="#1f77b4", width=2),
        ))
        fig_beta.update_layout(
            title=f"噪声调度 $\\beta_t$（{schedule_type}）",
            xaxis_title="步数 $t$", yaxis_title="$\\beta_t$",
            height=350,
        )
        chart.display_chart(fig_beta)

    # SNR 曲线
    fig_snr = go.Figure()
    fig_snr.add_trace(go.Scatter(
        x=steps_arr, y=snr, mode="lines",
        name="SNR", line=dict(color="#2ca02c", width=2),
    ))
    fig_snr.add_trace(go.Scatter(
        x=steps_arr, y=alphas_cumprod, mode="lines",
        name="$\\bar{\\alpha}_t$", line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig_snr.update_layout(
        title="信噪比 (SNR) 与 $\\bar{\\alpha}_t$ 随步数变化",
        xaxis_title="步数 $t$", yaxis_title="数值",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    chart.display_chart(fig_snr)

    st.markdown(f"""
    **关键指标**：
    - 初始 SNR: {snr[0]:.2f}（信号远强于噪声）
    - 最终 SNR: {snr[-1]:.4f}（噪声完全主导）
    - """)
    display_latex(rf"\bar{{\alpha}}_T = {alphas_cumprod[-1]:.6f}")
    st.markdown("（几乎为 0，信息完全丢失）")

    st.markdown("### 4.4 扩散过程可视化")

    st.markdown("展示数据在不同扩散步骤下逐渐变为纯噪声的过程（以同心圆图案为例）：")

    images = generate_diffusion_images(diff_T)

    n_images = len(images)
    cols_vis = st.columns(min(n_images, 7))

    for idx, (t, img) in enumerate(images.items()):
        with cols_vis[idx % len(cols_vis)]:
            fig_img = go.Figure(data=go.Heatmap(
                z=img, colorscale="gray", showscale=False, zmin=-2, zmax=2,
            ))
            noise_ratio = t / max(diff_T - 1, 1)
            fig_img.update_layout(
                title=f"$t={t}$<br>噪声: {noise_ratio:.0%}",
                height=160, margin=dict(l=10, r=10, t=40, b=10),
            )
            chart.display_chart(fig_img)

    # ============================================================
    # Section 5: Model Comparison 模型对比
    # ============================================================
    st.markdown("---")
    st.markdown("## 5. Model Comparison 模型对比")

    st.markdown("### 5.1 生成模型综合对比")

    st.markdown("基于论文基准数据，对比四种主流生成模型在关键指标上的表现：")

    # 参考数据（基于论文报告的典型范围，归一化到 0-10 分制）
    models = ["Autoencoder", "VAE", "GAN", "Diffusion"]
    metrics_names = ["训练稳定性", "样本质量", "模式覆盖", "推理速度"]

    # 训练稳定性 (0-10)
    train_stability = [8.5, 8.0, 4.0, 9.0]
    # 样本质量 (0-10, 基于 FID 分数换算)
    sample_quality = [4.0, 5.5, 8.0, 9.5]
    # 模式覆盖 (0-10)
    mode_coverage = [3.0, 6.0, 5.0, 9.0]
    # 推理速度 (0-10, 越高越快)
    inference_speed = [9.0, 8.5, 8.0, 2.0]

    fig_compare = go.Figure()

    colors_compare = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    x_positions = np.arange(len(metrics_names))
    bar_width = 0.18

    for i, model in enumerate(models):
        values = [train_stability[i], sample_quality[i], mode_coverage[i], inference_speed[i]]
        offset = (i - 1.5) * bar_width
        fig_compare.add_trace(go.Bar(
            x=x_positions + offset,
            y=values,
            name=model,
            marker_color=colors_compare[i],
            width=bar_width,
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
            textfont=dict(size=9),
        ))

    fig_compare.update_layout(
        title="生成模型多维度对比（分数 0-10，越高越好）",
        xaxis=dict(
            tickmode="array",
            tickvals=x_positions,
            ticktext=metrics_names,
        ),
        yaxis=dict(title="评分", range=[0, 11]),
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    chart.display_chart(fig_compare)

    st.markdown("### 5.2 详细指标说明")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        **训练稳定性 Training Stability**

        | 模型 | 评分 | 说明 |
        |------|------|------|
        | Diffusion | 9.0 | 稳定的对数似然目标，无对抗训练 |
        | Autoencoder | 8.5 | 简单的重建损失，非常稳定 |
        | VAE | 8.0 | ELBO 目标稳定，但需平衡重建与 KL |
        | GAN | 4.0 | 对抗训练不稳定，模式崩塌风险高 |

        **样本质量 Sample Quality**（基于 FID 分数）

        | 模型 | 评分 | 说明 |
        |------|------|------|
        | Diffusion | 9.5 | SOTA 生成质量 (FID < 3.0 on CIFAR-10) |
        | GAN | 8.0 | 高质量但多样性不足 (StyleGAN2) |
        | VAE | 5.5 | 生成样本偏模糊 |
        | Autoencoder | 4.0 | 无法生成新样本 |
        """)

    with col_c2:
        st.markdown("""
        **模式覆盖 Mode Coverage**

        | 模型 | 评分 | 说明 |
        |------|------|------|
        | Diffusion | 9.0 | 覆盖数据分布的所有模式 |
        | VAE | 6.0 | KL 正则化促进模式覆盖 |
        | GAN | 5.0 | 容易遗漏低概率模式 |
        | Autoencoder | 3.0 | 潜在空间不连续，无法插值 |

        **推理速度 Inference Speed**

        | 模型 | 评分 | 说明 |
        |------|------|------|
        | Autoencoder | 9.0 | 单次前向传播，极快 |
        | VAE | 8.5 | 单次前向传播 + 采样 |
        | GAN | 8.0 | 单次前向传播 |
        | Diffusion | 2.0 | 需要数百到数千步迭代去噪 |
        """)

    st.markdown("### 5.3 适用场景推荐")

    st.markdown("""
    | 场景 | 推荐模型 | 原因 |
    |------|----------|------|
    | 特征提取 / 降维 | Autoencoder | 简单高效，无需生成新样本 |
    | 生成 + 表示学习 | VAE | 潜在空间结构良好，支持插值 |
    | 高质量图像生成 | GAN / Diffusion | GAN 推理快，Diffusion 质量高 |
    | 条件生成 / 控制 | Diffusion | Classifier-Free Guidance 提供精确控制 |
    | 实时应用 | GAN | 单步生成，延迟最低 |
    | 训练资源有限 | VAE | 训练最稳定，对超参数不敏感 |
    """)

    st.info("""
    **总结**：
    - 没有"最好的"生成模型，选择取决于具体需求
    - **质量优先** → Diffusion Model
    - **速度优先** → GAN
    - **稳定性优先** → VAE
    - **表示学习优先** → VAE / Autoencoder
    - 实际应用中常结合多种方法，如 VAE+GAN (VAE-GAN)、Diffusion+GAN 加速采样等
    """)
