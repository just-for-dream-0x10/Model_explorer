"""
扩散模型（Diffusion Models）数学原理模块

包含：
- DDPM (Denoising Diffusion Probabilistic Models)
- 前向扩散过程与反向去噪过程
- 数学推导和可视化
- 图像生成演示
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from simple_latex import display_latex
from PIL import Image
import torchvision.transforms as transforms


class SimpleUNet(nn.Module):
    """简化版U-Net用于扩散模型去噪"""
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 编码器
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        # 解码器
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, out_channels, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def pos_encoding(self, t, channels):
        """位置编码用于时间步"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        # 时间嵌入
        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # 编码
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.conv2(x2))
        x3 = self.pool(x2)
        x3 = F.relu(self.conv3(x3))
        
        # 解码
        x = self.upsample(x3)
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        
        return x


def get_beta_schedule(schedule_name, timesteps, beta_start=0.0001, beta_end=0.02):
    """
    获取不同的噪声调度策略
    
    参数:
        schedule_name: 'linear', 'cosine', 'quadratic'
        timesteps: 扩散步数
        beta_start: 起始β值
        beta_end: 结束β值
    """
    if schedule_name == 'linear':
        return np.linspace(beta_start, beta_end, timesteps)
    elif schedule_name == 'cosine':
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    elif schedule_name == 'quadratic':
        return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


class DiffusionModel:
    """DDPM扩散模型实现"""
    
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps
        
        # 计算beta和alpha
        self.betas = get_beta_schedule(beta_schedule, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # 用于q(x_t | x_0)的计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于后验q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 调整维度以便广播
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[..., np.newaxis]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[..., np.newaxis]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, device='cpu'):
        """
        反向去噪过程：p(x_{t-1} | x_t)
        使用模型预测的噪声来恢复x_{t-1}
        """
        # 预测噪声
        with torch.no_grad():
            x_t_tensor = torch.FloatTensor(x_t).to(device)
            t_tensor = torch.LongTensor([t]).to(device)
            predicted_noise = model(x_t_tensor.unsqueeze(0).unsqueeze(0), t_tensor)
            predicted_noise = predicted_noise.squeeze().cpu().numpy()
        
        # 计算x_0的估计
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        x_0_pred = (x_t - np.sqrt(1 - alpha_cumprod_t) * predicted_noise) / np.sqrt(alpha_cumprod_t)
        
        # 计算均值
        if t > 0:
            posterior_variance_t = self.posterior_variance[t]
            noise = np.random.randn(*x_t.shape)
            x_prev = (
                np.sqrt(alpha_t) * (1 - self.alphas_cumprod_prev[t]) / (1 - alpha_cumprod_t) * x_t +
                np.sqrt(self.alphas_cumprod_prev[t]) * beta_t / (1 - alpha_cumprod_t) * x_0_pred +
                np.sqrt(posterior_variance_t) * noise
            )
        else:
            x_prev = x_0_pred
        
        return x_prev, x_0_pred


def create_2d_gaussian_data(n_samples=500, noise_level=0.1):
    """创建2D高斯混合数据用于可视化"""
    # 创建Swiss Roll或其他2D分布
    theta = np.sqrt(np.random.rand(n_samples)) * 3 * np.pi
    r = 2 * theta + np.random.randn(n_samples) * noise_level
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def diffusion_tab(CHINESE_SUPPORTED):
    """扩散模型主标签页"""
    
    if CHINESE_SUPPORTED:
        st.title("🌊 扩散模型（Diffusion Models）数学原理")
        st.markdown("""
        扩散模型是当前最先进的生成模型之一，广泛应用于图像生成（Stable Diffusion、DALL-E）、
        音频合成、视频生成等领域。本模块深入探讨其数学原理。
        """)
    else:
        st.title("🌊 Diffusion Models: Mathematical Principles")
        st.markdown("""
        Diffusion models are state-of-the-art generative models widely used in image generation 
        (Stable Diffusion, DALL-E), audio synthesis, and video generation.
        """)
    
    # 侧边栏参数
    st.sidebar.header("扩散模型参数" if CHINESE_SUPPORTED else "Diffusion Parameters")
    
    timesteps = st.sidebar.slider(
        "扩散步数 (T)" if CHINESE_SUPPORTED else "Timesteps (T)",
        min_value=50, max_value=1000, value=200, step=50
    )
    
    beta_schedule = st.sidebar.selectbox(
        "β调度策略" if CHINESE_SUPPORTED else "Beta Schedule",
        ['linear', 'cosine', 'quadratic']
    )
    
    # 生成可视化时间步选项
    timestep_options = list(range(0, timesteps, max(1, timesteps//10)))
    if timesteps - 1 not in timestep_options:
        timestep_options.append(timesteps - 1)
    
    # 确保默认值在选项中
    default_steps = []
    for t in [0, timesteps//4, timesteps//2, 3*timesteps//4, timesteps-1]:
        # 找到最接近的选项
        closest = min(timestep_options, key=lambda x: abs(x - t))
        if closest not in default_steps:
            default_steps.append(closest)
    
    visualization_timesteps = st.sidebar.multiselect(
        "可视化时间步" if CHINESE_SUPPORTED else "Visualization Timesteps",
        timestep_options,
        default=default_steps[:5]  # 最多5个默认值
    )
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 数学理论" if CHINESE_SUPPORTED else "📚 Theory",
        "➡️ 前向扩散" if CHINESE_SUPPORTED else "➡️ Forward Process",
        "⬅️ 反向去噪" if CHINESE_SUPPORTED else "⬅️ Reverse Process",
        "🎨 图像生成" if CHINESE_SUPPORTED else "🎨 Image Generation",
        "📊 2D可视化" if CHINESE_SUPPORTED else "📊 2D Visualization"
    ])
    
    # 创建扩散模型实例
    diffusion = DiffusionModel(timesteps=timesteps, beta_schedule=beta_schedule)
    
    with tab1:
        show_theory(CHINESE_SUPPORTED, diffusion)
    
    with tab2:
        show_forward_process(CHINESE_SUPPORTED, diffusion, visualization_timesteps)
    
    with tab3:
        show_reverse_process(CHINESE_SUPPORTED, diffusion, visualization_timesteps)
    
    with tab4:
        show_image_generation(CHINESE_SUPPORTED, diffusion, timesteps)
    
    with tab5:
        show_2d_visualization(CHINESE_SUPPORTED, diffusion, visualization_timesteps)


def show_theory(CHINESE_SUPPORTED, diffusion):
    """显示数学理论部分"""
    
    if CHINESE_SUPPORTED:
        st.header("扩散模型数学基础")
        
        st.subheader("1️⃣ 核心思想")
        st.markdown("""
        扩散模型通过两个过程生成数据：
        - **前向过程**：逐步向数据添加高斯噪声，直到变成纯噪声
        - **反向过程**：训练神经网络学习逆向去噪，从噪声恢复数据
        """)
        
        st.subheader("2️⃣ 前向扩散过程（Forward Process）")
        st.markdown("**马尔可夫链定义**：给定数据 $x_0 \\sim q(x_0)$，通过 $T$ 步逐渐添加噪声")
        
        display_latex(r"q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)")
        
        st.markdown("其中 $\\beta_t \\in (0,1)$ 是噪声调度表")
        
        st.markdown("**重要性质**：可以直接从 $x_0$ 采样任意时刻 $x_t$（重参数化技巧）")
        
        display_latex(r"q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)")
        
        st.markdown("其中：")
        display_latex(r"\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s")
        
        st.markdown("**采样公式**（重参数化）：")
        display_latex(r"x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)")
        
        st.subheader("3️⃣ 反向去噪过程（Reverse Process）")
        st.markdown("目标：学习反向转移概率 $p_\\theta(x_{t-1} | x_t)$")
        
        display_latex(r"p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))")
        
        st.markdown("**关键洞察**：当 $\\beta_t$ 足够小时，反向过程也是高斯分布！")
        
        st.subheader("4️⃣ 训练目标：变分下界（ELBO）")
        st.markdown("优化负对数似然的变分下界：")
        
        display_latex(r"\mathcal{L} = \mathbb{E}_q \left[ -\log p_\theta(x_0) \right] \leq \mathcal{L}_{\text{VLB}}")
        
        st.markdown("**简化目标**（Ho et al. 2020 DDPM）：")
        display_latex(r"\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]")
        
        st.markdown("即：训练神经网络 $\\epsilon_\\theta$ 预测添加的噪声！")
        
        st.subheader("5️⃣ 后验分布推导")
        st.markdown("真实的后验分布（使用贝叶斯定理）：")
        
        display_latex(r"q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)")
        
        st.markdown("其中均值为：")
        display_latex(r"\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t")
        
        st.markdown("方差为：")
        display_latex(r"\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t")
        
        st.subheader("6️⃣ 采样算法（DDPM）")
        st.code("""
# 从纯噪声开始
x_T ~ N(0, I)

# 逐步去噪
for t = T, T-1, ..., 1:
    # 预测噪声
    ε_θ = model(x_t, t)
    
    # 计算均值
    μ_θ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
    
    # 添加噪声（t>1时）
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = μ_θ + √β_t * z

return x_0
        """, language="python")
        
    else:
        st.header("Diffusion Models: Mathematical Foundations")
        # 英文版本
        st.subheader("Core Idea")
        st.markdown("""
        Diffusion models generate data through two processes:
        - **Forward Process**: Gradually add Gaussian noise until data becomes pure noise
        - **Reverse Process**: Train neural network to denoise and recover data
        """)
    
    # 显示β调度
    st.subheader("📊 Beta调度可视化" if CHINESE_SUPPORTED else "📊 Beta Schedule Visualization")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'β_t (Noise Schedule)', 
            'α_t = 1 - β_t',
            'ᾱ_t = ∏α_s (Cumulative Product)',
            '√ᾱ_t and √(1-ᾱ_t)'
        )
    )
    
    t_range = np.arange(len(diffusion.betas))
    
    # β_t
    fig.add_trace(go.Scatter(x=t_range, y=diffusion.betas, name='β_t', 
                             line=dict(color='red')), row=1, col=1)
    
    # α_t
    fig.add_trace(go.Scatter(x=t_range, y=diffusion.alphas, name='α_t',
                             line=dict(color='blue')), row=1, col=2)
    
    # ᾱ_t
    fig.add_trace(go.Scatter(x=t_range, y=diffusion.alphas_cumprod, name='ᾱ_t',
                             line=dict(color='green')), row=2, col=1)
    
    # 信号和噪声比例
    fig.add_trace(go.Scatter(x=t_range, y=diffusion.sqrt_alphas_cumprod, 
                             name='√ᾱ_t (signal)', line=dict(color='purple')), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_range, y=diffusion.sqrt_one_minus_alphas_cumprod,
                             name='√(1-ᾱ_t) (noise)', line=dict(color='orange')), row=2, col=2)
    
    fig.update_xaxes(title_text="Time step t")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 关键洞察
    if CHINESE_SUPPORTED:
        st.info("""
        **关键洞察**：
        - 当 t→T 时，√ᾱ_t → 0，√(1-ᾱ_t) → 1，数据完全变成噪声
        - 不同的β调度策略影响扩散速度
        - cosine调度在早期保留更多信息，后期更激进
        """)
    
    # 数学证明展开
    with st.expander("📖 重参数化公式证明" if CHINESE_SUPPORTED else "📖 Reparameterization Proof"):
        if CHINESE_SUPPORTED:
            st.markdown("**证明**：从 $q(x_t | x_0)$ 可以直接采样")
            st.markdown("从递推关系出发：")
            display_latex(r"x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}")
            st.markdown("递归展开：")
            display_latex(r"x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1}")
            st.markdown("利用高斯分布的性质，两个独立高斯噪声的和仍是高斯：")
            display_latex(r"\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)")
            st.markdown("继续递归到 $x_0$，最终得到：")
            display_latex(r"x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon")
            st.markdown("其中 $\\epsilon \\sim \\mathcal{N}(0, I)$")
        else:
            st.markdown("Proof of direct sampling from q(x_t | x_0)")
    
    with st.expander("📖 后验分布推导" if CHINESE_SUPPORTED else "📖 Posterior Derivation"):
        if CHINESE_SUPPORTED:
            st.markdown("**目标**：计算 $q(x_{t-1} | x_t, x_0)$")
            st.markdown("使用贝叶斯定理：")
            display_latex(r"q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)}")
            st.markdown("由马尔可夫性质，$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$")
            st.markdown("代入高斯分布形式，完成配方后得到后验均值和方差")
        else:
            st.markdown("Posterior q(x_{t-1} | x_t, x_0) derivation using Bayes' theorem")


def show_forward_process(CHINESE_SUPPORTED, diffusion, visualization_timesteps):
    """显示前向扩散过程"""
    
    if CHINESE_SUPPORTED:
        st.header("前向扩散过程：逐步添加噪声")
        st.markdown("""
        前向过程通过公式 $x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon$ 
        将原始图像逐步转化为纯噪声。
        """)
    else:
        st.header("Forward Diffusion: Gradually Adding Noise")
    
    # 选择图像类型
    image_type = st.selectbox(
        "选择输入图像" if CHINESE_SUPPORTED else "Select Input Image",
        ["简单图案", "数字图案", "渐变图案"] if CHINESE_SUPPORTED else 
        ["Simple Pattern", "Digit Pattern", "Gradient Pattern"]
    )
    
    # 创建原始图像
    size = 64
    if "简单" in image_type or "Simple" in image_type:
        # 创建一个简单的圆形图案
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 4
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        x_0 = mask.astype(float)
    elif "数字" in image_type or "Digit" in image_type:
        # 创建数字"8"的图案
        x_0 = np.zeros((size, size))
        # 上圆
        y, x = np.ogrid[:size, :size]
        center_y1, center_x = size // 3, size // 2
        radius = size // 6
        mask1 = (x - center_x)**2 + (y - center_y1)**2 <= radius**2
        # 下圆
        center_y2 = 2 * size // 3
        mask2 = (x - center_x)**2 + (y - center_y2)**2 <= radius**2
        x_0[mask1 | mask2] = 1.0
    else:
        # 渐变图案
        x_0 = np.linspace(0, 1, size)
        x_0 = np.outer(x_0, x_0)
    
    # 标准化到[-1, 1]
    x_0 = (x_0 - 0.5) * 2
    
    # 生成不同时间步的噪声图像
    if not visualization_timesteps:
        visualization_timesteps = [0, len(diffusion.betas)//4, len(diffusion.betas)//2, 
                                   3*len(diffusion.betas)//4, len(diffusion.betas)-1]
    
    num_vis = len(visualization_timesteps)
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=(num_vis + 1) // 2,
        subplot_titles=[f"t = {t}" for t in visualization_timesteps],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    noisy_images = []
    for idx, t in enumerate(visualization_timesteps):
        # 生成噪声图像
        noise = np.random.randn(*x_0.shape)
        x_t = diffusion.q_sample(x_0, np.array([t]), noise)
        noisy_images.append(x_t)
        
        row = idx // ((num_vis + 1) // 2) + 1
        col = idx % ((num_vis + 1) // 2) + 1
        
        fig.add_trace(
            go.Heatmap(z=x_t, colorscale='RdBu_r', zmid=0, 
                      showscale=(idx == num_vis - 1),
                      zmin=-3, zmax=3),
            row=row, col=col
        )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=400, title_text="不同时间步的扩散结果" if CHINESE_SUPPORTED else "Diffusion at Different Timesteps")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示数值统计
    if CHINESE_SUPPORTED:
        st.subheader("📊 数值统计")
        
        stats_data = []
        for idx, t in enumerate(visualization_timesteps):
            x_t = noisy_images[idx]
            signal_coef = diffusion.sqrt_alphas_cumprod[t]
            noise_coef = diffusion.sqrt_one_minus_alphas_cumprod[t]
            
            stats_data.append({
                "时间步 t": t,
                "√ᾱ_t (信号系数)": f"{signal_coef:.4f}",
                "√(1-ᾱ_t) (噪声系数)": f"{noise_coef:.4f}",
                "图像均值": f"{x_t.mean():.4f}",
                "图像标准差": f"{x_t.std():.4f}",
                "信噪比": f"{signal_coef/noise_coef:.4f}" if noise_coef > 0.01 else "∞"
            })
        
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)
        
        st.info("""
        **观察**：
        - 随着 t 增大，信号系数 √ᾱ_t 递减，噪声系数 √(1-ᾱ_t) 递增
        - 最终图像变成标准正态分布（均值≈0，标准差≈1）
        - 信噪比持续下降，直到信号完全被噪声淹没
        """)
    
    # 交互式单步演示
    with st.expander("🔍 单步扩散详细过程" if CHINESE_SUPPORTED else "🔍 Single Step Detailed Process"):
        single_t = st.slider(
            "选择时间步" if CHINESE_SUPPORTED else "Select Timestep",
            min_value=0,
            max_value=len(diffusion.betas) - 1,
            value=len(diffusion.betas) // 2
        )
        
        noise = np.random.randn(*x_0.shape)
        x_t = diffusion.q_sample(x_0, np.array([single_t]), noise)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**原始图像** $x_0$")
            fig1 = go.Figure(data=go.Heatmap(z=x_0, colorscale='RdBu_r', zmid=0))
            fig1.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            fig1.update_xaxes(showticklabels=False)
            fig1.update_yaxes(showticklabels=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("**纯噪声** $\\epsilon$")
            fig2 = go.Figure(data=go.Heatmap(z=noise, colorscale='RdBu_r', zmid=0))
            fig2.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown(f"**加噪结果** $x_{{{single_t}}}$")
            fig3 = go.Figure(data=go.Heatmap(z=x_t, colorscale='RdBu_r', zmid=0))
            fig3.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        if CHINESE_SUPPORTED:
            st.markdown("**计算公式**：")
            signal_coef = diffusion.sqrt_alphas_cumprod[single_t]
            noise_coef = diffusion.sqrt_one_minus_alphas_cumprod[single_t]
            st.latex(f"x_{{{single_t}}} = {signal_coef:.4f} \\cdot x_0 + {noise_coef:.4f} \\cdot \\epsilon")
            
            st.markdown(f"""
            - 信号保留比例：{signal_coef:.2%}
            - 噪声添加比例：{noise_coef:.2%}
            - β_{single_t} = {diffusion.betas[single_t]:.6f}
            - α_{single_t} = {diffusion.alphas[single_t]:.6f}
            - ᾱ_{single_t} = {diffusion.alphas_cumprod[single_t]:.6f}
            """)


def show_reverse_process(CHINESE_SUPPORTED, diffusion, visualization_timesteps):
    """显示反向去噪过程"""
    
    if CHINESE_SUPPORTED:
        st.header("反向去噪过程：从噪声恢复数据")
        st.markdown("""
        反向过程使用训练好的神经网络 $\\epsilon_\\theta(x_t, t)$ 预测噪声，
        然后通过公式逐步去噪恢复原始图像。
        """)
    else:
        st.header("Reverse Denoising: Recovering Data from Noise")
    
    st.info("""
    **注意**：完整训练扩散模型需要大量数据和计算资源。这里我们演示反向过程的数学原理，
    使用理想情况（已知真实噪声）来展示去噪过程。
    """ if CHINESE_SUPPORTED else """
    **Note**: Training a full diffusion model requires substantial data and computation.
    Here we demonstrate the mathematical principles using the ideal case (known noise).
    """)
    
    # 创建原始图像
    size = 64
    image_type = st.selectbox(
        "选择目标图像" if CHINESE_SUPPORTED else "Select Target Image",
        ["圆形", "方形", "三角形", "星形"] if CHINESE_SUPPORTED else 
        ["Circle", "Square", "Triangle", "Star"],
        key="reverse_image_type"
    )
    
    x_0 = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    if "圆形" in image_type or "Circle" in image_type:
        radius = size // 4
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        x_0[mask] = 1.0
    elif "方形" in image_type or "Square" in image_type:
        x_0[size//4:3*size//4, size//4:3*size//4] = 1.0
    elif "三角形" in image_type or "Triangle" in image_type:
        for i in range(size):
            for j in range(size):
                if i >= size//4 and i <= 3*size//4:
                    width = (i - size//4) * size // (2*size)
                    if abs(j - center) <= width:
                        x_0[i, j] = 1.0
    else:  # 星形
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        for angle in angles:
            for r in np.linspace(0, size//4, 20):
                px = int(center + r * np.cos(angle))
                py = int(center + r * np.sin(angle))
                if 0 <= px < size and 0 <= py < size:
                    x_0[py, px] = 1.0
    
    x_0 = (x_0 - 0.5) * 2  # 标准化到[-1, 1]
    
    # 选择演示类型
    demo_type = st.radio(
        "演示类型" if CHINESE_SUPPORTED else "Demo Type",
        ["理想去噪（已知真实噪声）", "逐步去噪过程"] if CHINESE_SUPPORTED else
        ["Ideal Denoising (Known Noise)", "Step-by-Step Denoising"]
    )
    
    if "理想" in demo_type or "Ideal" in demo_type:
        # 理想情况：我们知道真实噪声
        st.subheader("理想去噪演示" if CHINESE_SUPPORTED else "Ideal Denoising Demo")
        
        # 生成加噪图像
        t = st.slider(
            "噪声水平 (t)" if CHINESE_SUPPORTED else "Noise Level (t)",
            min_value=1,
            max_value=len(diffusion.betas) - 1,
            value=len(diffusion.betas) // 2,
            key="ideal_t"
        )
        
        noise = np.random.randn(*x_0.shape)
        x_t = diffusion.q_sample(x_0, np.array([t]), noise)
        
        # 使用已知噪声"去噪"（理想情况）
        alpha_cumprod_t = diffusion.alphas_cumprod[t]
        x_0_pred = (x_t - np.sqrt(1 - alpha_cumprod_t) * noise) / np.sqrt(alpha_cumprod_t)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**原始图像** $x_0$")
            fig1 = go.Figure(data=go.Heatmap(z=x_0, colorscale='RdBu_r', zmid=0, zmin=-2, zmax=2))
            fig1.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            fig1.update_xaxes(showticklabels=False)
            fig1.update_yaxes(showticklabels=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown(f"**加噪图像** $x_{{{t}}}$")
            fig2 = go.Figure(data=go.Heatmap(z=x_t, colorscale='RdBu_r', zmid=0, zmin=-2, zmax=2))
            fig2.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            fig2.update_xaxes(showticklabels=False)
            fig2.update_yaxes(showticklabels=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown("**恢复图像** $\\hat{x}_0$")
            fig3 = go.Figure(data=go.Heatmap(z=x_0_pred, colorscale='RdBu_r', zmid=0, zmin=-2, zmax=2))
            fig3.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            fig3.update_xaxes(showticklabels=False)
            fig3.update_yaxes(showticklabels=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        # 计算误差
        mse = np.mean((x_0_pred - x_0)**2)
        if CHINESE_SUPPORTED:
            st.markdown(f"""
            **去噪公式**：
            $$\\hat{{x}}_0 = \\frac{{x_t - \\sqrt{{1-\\bar{{\\alpha}}_t}} \\cdot \\epsilon}}{{\\sqrt{{\\bar{{\\alpha}}_t}}}}$$
            
            **重建误差 (MSE)**: {mse:.6f}
            
            在理想情况下（已知真实噪声），我们可以几乎完美地恢复原始图像。
            实际应用中，神经网络需要学习预测这个噪声。
            """)
    
    else:
        # 逐步去噪过程
        st.subheader("逐步去噪可视化" if CHINESE_SUPPORTED else "Step-by-Step Denoising")
        
        # 从完全噪声开始
        num_steps = st.slider(
            "去噪步数" if CHINESE_SUPPORTED else "Denoising Steps",
            min_value=5, max_value=50, value=10, step=5
        )
        
        # 生成完全噪声
        x_t = np.random.randn(*x_0.shape)
        
        # 选择时间步
        timesteps_to_show = np.linspace(len(diffusion.betas)-1, 0, num_steps).astype(int)
        
        # 模拟去噪过程（使用已知的x_0）
        denoising_steps = []
        current_x = x_t.copy()
        
        for t in timesteps_to_show:
            # 计算x_0的估计（在真实场景中，这来自神经网络预测的噪声）
            if t > 0:
                # 模拟：假设我们能预测噪声
                noise_pred = (current_x - np.sqrt(diffusion.alphas_cumprod[t]) * x_0) / np.sqrt(1 - diffusion.alphas_cumprod[t])
                
                # 计算x_{t-1}
                alpha_t = diffusion.alphas[t]
                alpha_cumprod_t = diffusion.alphas_cumprod[t]
                beta_t = diffusion.betas[t]
                
                x_0_pred = (current_x - np.sqrt(1 - alpha_cumprod_t) * noise_pred) / np.sqrt(alpha_cumprod_t)
                
                # 后验均值
                posterior_mean = (
                    np.sqrt(diffusion.alphas_cumprod[t-1]) * beta_t / (1 - alpha_cumprod_t) * x_0_pred +
                    np.sqrt(alpha_t) * (1 - diffusion.alphas_cumprod[t-1]) / (1 - alpha_cumprod_t) * current_x
                )
                
                # 添加噪声
                posterior_variance = diffusion.posterior_variance[t]
                noise = np.random.randn(*x_0.shape)
                current_x = posterior_mean + np.sqrt(posterior_variance) * noise
            else:
                current_x = (current_x - np.sqrt(1 - diffusion.alphas_cumprod[t]) * 
                           (current_x - np.sqrt(diffusion.alphas_cumprod[t]) * x_0) / 
                           np.sqrt(1 - diffusion.alphas_cumprod[t])) / np.sqrt(diffusion.alphas_cumprod[t])
            
            denoising_steps.append((t, current_x.copy()))
        
        # 显示去噪过程
        num_vis = min(6, len(denoising_steps))
        indices = np.linspace(0, len(denoising_steps)-1, num_vis).astype(int)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"t = {denoising_steps[i][0]}" for i in indices],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, i in enumerate(indices):
            t, img = denoising_steps[i]
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            fig.add_trace(
                go.Heatmap(z=img, colorscale='RdBu_r', zmid=0, 
                          showscale=(idx == num_vis - 1),
                          zmin=-2, zmax=2),
                row=row, col=col
            )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=500, title_text="去噪过程" if CHINESE_SUPPORTED else "Denoising Process")
        
        st.plotly_chart(fig, use_container_width=True)
        
        if CHINESE_SUPPORTED:
            st.info("""
            **观察**：
            - 从纯噪声开始，逐步出现图像结构
            - 早期步骤确定大致轮廓，后期步骤细化细节
            - 这个过程模拟了训练好的扩散模型的采样过程
            """)


def show_image_generation(CHINESE_SUPPORTED, diffusion, timesteps):
    """显示图像生成演示"""
    
    if CHINESE_SUPPORTED:
        st.header("图像生成演示")
        st.markdown("""
        这里演示如何使用扩散模型生成图像。由于完整训练需要大量资源，
        我们展示采样算法的工作原理，以及不同采样方法的对比。
        """)
    else:
        st.header("Image Generation Demo")
    
    # 选择采样方法
    sampling_method = st.selectbox(
        "采样方法" if CHINESE_SUPPORTED else "Sampling Method",
        ["DDPM (标准采样)", "DDIM (加速采样)", "对比展示"] if CHINESE_SUPPORTED else
        ["DDPM (Standard)", "DDIM (Accelerated)", "Comparison"]
    )
    
    if "对比" in sampling_method or "Comparison" in sampling_method:
        show_sampling_comparison(CHINESE_SUPPORTED, diffusion)
    else:
        show_single_sampling(CHINESE_SUPPORTED, diffusion, sampling_method)
    
    # 算法详解
    with st.expander("📖 DDPM vs DDIM 算法对比" if CHINESE_SUPPORTED else "📖 DDPM vs DDIM Algorithm Comparison"):
        if CHINESE_SUPPORTED:
            st.markdown("### DDPM (Denoising Diffusion Probabilistic Models)")
            st.markdown("""
            **特点**：
            - 马尔可夫采样过程，需要遍历所有时间步
            - 每步添加随机噪声，生成结果具有随机性
            - 高质量但速度较慢（T步采样）
            """)
            
            st.markdown("**采样公式**：")
            display_latex(r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z")
            
            st.markdown("其中 $z \\sim \\mathcal{N}(0, I)$，$\\sigma_t = \\sqrt{\\beta_t}$")
            
            st.markdown("---")
            
            st.markdown("### DDIM (Denoising Diffusion Implicit Models)")
            st.markdown("""
            **特点**：
            - 非马尔可夫过程，可以跳过时间步
            - 确定性采样（η=0时），相同初始噪声生成相同结果
            - 速度快（可用10-50步代替1000步）
            """)
            
            st.markdown("**采样公式**：")
            display_latex(r"x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t z")
            
            st.markdown("其中 $\\sigma_t = \\eta \\sqrt{(1-\\bar{\\alpha}_{t-1})/(1-\\bar{\\alpha}_t)} \\sqrt{1-\\bar{\\alpha}_t/\\bar{\\alpha}_{t-1}}$")
            st.markdown("- $\\eta = 0$: 完全确定性")
            st.markdown("- $\\eta = 1$: 等价于DDPM")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**DDPM优势**")
                st.markdown("- 理论保证强")
                st.markdown("- 样本质量高")
                st.markdown("- 多样性好")
            
            with col2:
                st.markdown("**DDIM优势**")
                st.markdown("- 采样速度快10-100倍")
                st.markdown("- 可确定性生成")
                st.markdown("- 支持插值和编辑")
        else:
            st.markdown("### DDPM (Denoising Diffusion Probabilistic Models)")
            st.markdown("- Markov sampling, slower but high quality")
            st.markdown("### DDIM (Denoising Diffusion Implicit Models)")
            st.markdown("- Non-Markov, 10-100x faster, deterministic option")
    
    # 实际应用
    with st.expander("🌟 扩散模型的实际应用" if CHINESE_SUPPORTED else "🌟 Real-world Applications"):
        if CHINESE_SUPPORTED:
            st.markdown("""
            ### 图像生成
            - **Stable Diffusion**: 开源文本到图像模型
            - **DALL-E 2/3**: OpenAI的图像生成系统
            - **Midjourney**: 艺术图像生成
            - **Imagen**: Google的高分辨率图像生成
            
            ### 其他应用
            - **视频生成**: Runway Gen-2, Pika
            - **音频合成**: DiffWave, WaveGrad
            - **3D生成**: DreamFusion, Point-E
            - **图像编辑**: InstructPix2Pix
            - **超分辨率**: SR3, Imagen Video
            - **医学影像**: 去噪、重建、分割
            
            ### 关键技术
            - **Classifier-Free Guidance**: 提高生成质量和可控性
            - **Latent Diffusion**: 在潜空间操作，降低计算成本
            - **ControlNet**: 精确控制生成过程
            - **LoRA**: 高效微调
            """)
        else:
            st.markdown("Applications: Stable Diffusion, DALL-E, Midjourney, video/audio generation, etc.")


def show_single_sampling(CHINESE_SUPPORTED, diffusion, method):
    """单个采样方法演示"""
    
    is_ddim = "DDIM" in method
    
    if CHINESE_SUPPORTED:
        st.subheader(f"{'DDIM' if is_ddim else 'DDPM'} 采样演示")
    else:
        st.subheader(f"{'DDIM' if is_ddim else 'DDPM'} Sampling Demo")
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        if is_ddim:
            num_inference_steps = st.slider(
                "推理步数" if CHINESE_SUPPORTED else "Inference Steps",
                min_value=10, max_value=100, value=50, step=10
            )
            eta = st.slider(
                "随机性参数 η" if CHINESE_SUPPORTED else "Stochasticity η",
                min_value=0.0, max_value=1.0, value=0.0, step=0.1
            )
        else:
            num_inference_steps = min(len(diffusion.betas), 200)
            eta = 1.0
    
    with col2:
        image_size = st.selectbox(
            "图像尺寸" if CHINESE_SUPPORTED else "Image Size",
            [32, 64],
            index=0
        )
    
    # 生成按钮
    if st.button("开始生成" if CHINESE_SUPPORTED else "Generate", key=f"gen_{method}"):
        with st.spinner("生成中..." if CHINESE_SUPPORTED else "Generating..."):
            # 创建简单目标图像（用于演示）
            target_images = create_target_images(image_size)
            
            # 选择一个目标
            target_idx = np.random.randint(0, len(target_images))
            x_0_target = target_images[target_idx]
            
            # 从噪声开始
            x_t = np.random.randn(image_size, image_size)
            
            # 选择时间步
            if is_ddim:
                timesteps_to_use = np.linspace(len(diffusion.betas)-1, 0, num_inference_steps).astype(int)
            else:
                timesteps_to_use = np.arange(len(diffusion.betas)-1, -1, -1)[:num_inference_steps]
            
            # 生成过程
            generated_steps = []
            for i, t in enumerate(timesteps_to_use):
                if i % max(1, len(timesteps_to_use) // 10) == 0:
                    generated_steps.append((t, x_t.copy()))
                
                # 模拟去噪（使用目标图像作为参考）
                if t > 0:
                    # 预测的噪声（简化：使用已知信息）
                    noise_pred = (x_t - np.sqrt(diffusion.alphas_cumprod[t]) * x_0_target) / np.sqrt(1 - diffusion.alphas_cumprod[t] + 1e-8)
                    
                    if is_ddim:
                        # DDIM采样
                        t_prev = timesteps_to_use[i+1] if i+1 < len(timesteps_to_use) else 0
                        alpha_t = diffusion.alphas_cumprod[t]
                        alpha_t_prev = diffusion.alphas_cumprod[t_prev] if t_prev > 0 else 1.0
                        
                        # 预测x_0
                        pred_x0 = (x_t - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
                        pred_x0 = np.clip(pred_x0, -2, 2)
                        
                        # 方向指向x_t
                        dir_xt = np.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise_pred
                        
                        # 随机项
                        sigma = eta * np.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_t_prev)
                        noise = np.random.randn(*x_t.shape) if eta > 0 else 0
                        
                        x_t = np.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
                    else:
                        # DDPM采样
                        alpha_t = diffusion.alphas[t]
                        alpha_cumprod_t = diffusion.alphas_cumprod[t]
                        beta_t = diffusion.betas[t]
                        
                        pred_x0 = (x_t - np.sqrt(1 - alpha_cumprod_t) * noise_pred) / np.sqrt(alpha_cumprod_t)
                        pred_x0 = np.clip(pred_x0, -2, 2)
                        
                        # 后验均值
                        alpha_cumprod_prev = diffusion.alphas_cumprod[t-1] if t > 0 else 1.0
                        posterior_mean = (
                            np.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t) * pred_x0 +
                            np.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x_t
                        )
                        
                        posterior_variance = diffusion.posterior_variance[t]
                        noise = np.random.randn(*x_t.shape)
                        x_t = posterior_mean + np.sqrt(posterior_variance) * noise
                else:
                    # 最后一步
                    pred_x0 = (x_t - np.sqrt(1 - diffusion.alphas_cumprod[0]) * noise_pred) / np.sqrt(diffusion.alphas_cumprod[0])
                    x_t = np.clip(pred_x0, -2, 2)
            
            # 显示结果
            st.success("生成完成！" if CHINESE_SUPPORTED else "Generation Complete!")
            
            # 显示生成过程
            num_vis = min(6, len(generated_steps))
            indices = np.linspace(0, len(generated_steps)-1, num_vis).astype(int)
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"t = {generated_steps[i][0]}" for i in indices],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            for idx, i in enumerate(indices):
                t, img = generated_steps[i]
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                fig.add_trace(
                    go.Heatmap(z=img, colorscale='RdBu_r', zmid=0, 
                              showscale=(idx == num_vis - 1),
                              zmin=-2, zmax=2),
                    row=row, col=col
                )
            
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(height=500, title_text="生成过程" if CHINESE_SUPPORTED else "Generation Process")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 最终结果
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**目标图像**" if CHINESE_SUPPORTED else "**Target Image**")
                fig_target = go.Figure(data=go.Heatmap(z=x_0_target, colorscale='RdBu_r', zmid=0))
                fig_target.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                fig_target.update_xaxes(showticklabels=False)
                fig_target.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_target, use_container_width=True)
            
            with col2:
                st.markdown("**生成结果**" if CHINESE_SUPPORTED else "**Generated Image**")
                fig_gen = go.Figure(data=go.Heatmap(z=x_t, colorscale='RdBu_r', zmid=0))
                fig_gen.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                fig_gen.update_xaxes(showticklabels=False)
                fig_gen.update_yaxes(showticklabels=False)
                st.plotly_chart(fig_gen, use_container_width=True)


def show_sampling_comparison(CHINESE_SUPPORTED, diffusion):
    """对比不同采样方法"""
    if CHINESE_SUPPORTED:
        st.subheader("DDPM vs DDIM 采样对比")
        st.markdown("直观比较两种采样方法的速度和质量差异")
    else:
        st.subheader("DDPM vs DDIM Comparison")
    
    # 参数
    ddim_steps = st.slider(
        "DDIM步数" if CHINESE_SUPPORTED else "DDIM Steps",
        min_value=10, max_value=100, value=20, step=10
    )
    
    ddpm_steps = st.slider(
        "DDPM步数" if CHINESE_SUPPORTED else "DDPM Steps",
        min_value=50, max_value=200, value=100, step=25
    )
    
    # 对比表格
    if CHINESE_SUPPORTED:
        comparison_data = {
            "特性": ["采样步数", "是否随机", "采样速度", "样本质量", "可控性"],
            "DDPM": [f"{ddpm_steps}步", "是", "慢", "高", "中等"],
            "DDIM": [f"{ddim_steps}步", "可选", "快", "高", "强"]
        }
    else:
        comparison_data = {
            "Feature": ["Steps", "Stochastic", "Speed", "Quality", "Control"],
            "DDPM": [f"{ddpm_steps}", "Yes", "Slow", "High", "Medium"],
            "DDIM": [f"{ddim_steps}", "Optional", "Fast", "High", "Strong"]
        }
    
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True)
    
    if CHINESE_SUPPORTED:
        st.info(f"""
        **速度提升**: DDIM使用{ddim_steps}步 vs DDPM使用{ddpm_steps}步
        - 理论加速比: {ddpm_steps/ddim_steps:.1f}x
        - 实际应用中，DDIM可以用50步达到DDPM 1000步的质量
        - Stable Diffusion默认使用DDIM的变体（DPM-Solver）
        """)


def create_target_images(size):
    """创建一些目标图像用于演示"""
    images = []
    
    # 圆形
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 4
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    img = np.zeros((size, size))
    img[mask] = 1.0
    images.append((img - 0.5) * 2)
    
    # 方形
    img = np.zeros((size, size))
    img[size//4:3*size//4, size//4:3*size//4] = 1.0
    images.append((img - 0.5) * 2)
    
    # 十字
    img = np.zeros((size, size))
    img[size//2-2:size//2+2, :] = 1.0
    img[:, size//2-2:size//2+2] = 1.0
    images.append((img - 0.5) * 2)
    
    return images


def show_2d_visualization(CHINESE_SUPPORTED, diffusion, visualization_timesteps):
    """显示2D数据扩散可视化"""
    
    if CHINESE_SUPPORTED:
        st.header("2D数据分布的扩散过程")
        st.markdown("""
        使用2D数据可视化扩散过程更加直观。我们可以看到数据分布如何从复杂的结构
        逐渐变成简单的高斯分布，以及反向过程如何恢复原始分布。
        """)
    else:
        st.header("Diffusion Process on 2D Data Distributions")
    
    # 选择数据分布类型
    data_type = st.selectbox(
        "选择数据分布" if CHINESE_SUPPORTED else "Select Data Distribution",
        ["Swiss Roll (瑞士卷)", "Two Moons (双月)", "Concentric Circles (同心圆)", 
         "Gaussian Mixture (高斯混合)"] if CHINESE_SUPPORTED else
        ["Swiss Roll", "Two Moons", "Concentric Circles", "Gaussian Mixture"]
    )
    
    n_samples = st.slider(
        "样本数量" if CHINESE_SUPPORTED else "Number of Samples",
        min_value=200, max_value=2000, value=500, step=100
    )
    
    # 生成数据
    if "Swiss Roll" in data_type or "瑞士卷" in data_type:
        theta = np.sqrt(np.random.rand(n_samples)) * 3 * np.pi
        r = 2 * theta + np.random.randn(n_samples) * 0.1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data = np.stack([x, y], axis=1)
    elif "Two Moons" in data_type or "双月" in data_type:
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        data = data * 2
    elif "Concentric" in data_type or "同心圆" in data_type:
        # 两个同心圆
        n_per_circle = n_samples // 2
        theta1 = np.random.rand(n_per_circle) * 2 * np.pi
        r1 = 1 + np.random.randn(n_per_circle) * 0.1
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        
        theta2 = np.random.rand(n_samples - n_per_circle) * 2 * np.pi
        r2 = 2.5 + np.random.randn(n_samples - n_per_circle) * 0.1
        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)
        
        data = np.stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])], axis=1)
    else:
        # 高斯混合
        centers = [[-2, -2], [2, 2], [-2, 2], [2, -2]]
        n_per_center = n_samples // len(centers)
        data_list = []
        for center in centers:
            samples = np.random.randn(n_per_center, 2) * 0.3 + center
            data_list.append(samples)
        data = np.vstack(data_list)
    
    # 标准化
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # 选择可视化时间步
    if not visualization_timesteps:
        visualization_timesteps = [0, len(diffusion.betas)//4, len(diffusion.betas)//2, 
                                   3*len(diffusion.betas)//4, len(diffusion.betas)-1]
    
    # 生成不同时间步的扩散数据
    diffused_data = {}
    for t in visualization_timesteps:
        noise = np.random.randn(*data.shape)
        x_t = diffusion.q_sample(data, np.array([t] * len(data)), noise)
        diffused_data[t] = x_t
    
    # 可视化
    num_vis = len(visualization_timesteps)
    cols_per_row = 3
    rows = (num_vis + cols_per_row - 1) // cols_per_row
    
    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=[f"t = {t}" for t in visualization_timesteps],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, t in enumerate(visualization_timesteps):
        row = idx // cols_per_row + 1
        col = idx % cols_per_row + 1
        
        x_t = diffused_data[t]
        
        fig.add_trace(
            go.Scatter(
                x=x_t[:, 0], y=x_t[:, 1],
                mode='markers',
                marker=dict(size=3, color=t, colorscale='Viridis', 
                           showscale=(idx == num_vis - 1)),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 设置相同的坐标范围
        fig.update_xaxes(range=[-4, 4], row=row, col=col)
        fig.update_yaxes(range=[-4, 4], row=row, col=col)
    
    fig.update_layout(height=300 * rows, title_text="2D数据的扩散过程" if CHINESE_SUPPORTED else "Diffusion Process on 2D Data")
    st.plotly_chart(fig, use_container_width=True)
    
    # 统计信息
    if CHINESE_SUPPORTED:
        st.subheader("📊 分布统计")
        
        stats = []
        for t in visualization_timesteps:
            x_t = diffused_data[t]
            stats.append({
                "时间步 t": t,
                "均值 (x)": f"{x_t[:, 0].mean():.4f}",
                "均值 (y)": f"{x_t[:, 1].mean():.4f}",
                "标准差 (x)": f"{x_t[:, 0].std():.4f}",
                "标准差 (y)": f"{x_t[:, 1].std():.4f}",
                "相关系数": f"{np.corrcoef(x_t[:, 0], x_t[:, 1])[0, 1]:.4f}"
            })
        
        df = pd.DataFrame(stats)
        st.dataframe(df, use_container_width=True)
        
        st.info("""
        **关键观察**：
        - 原始数据可能有复杂的结构和相关性
        - 随着扩散进行，数据逐渐失去结构，变成各向同性的高斯分布
        - 最终状态：均值≈0，标准差≈1，各维度独立（相关系数≈0）
        - 这就是为什么扩散模型能从简单的高斯噪声生成复杂数据！
        """)
    
    # 动画演示
    with st.expander("🎬 动画演示" if CHINESE_SUPPORTED else "🎬 Animation Demo"):
        if st.button("生成扩散动画" if CHINESE_SUPPORTED else "Generate Diffusion Animation"):
            # 创建更多时间步用于动画
            animation_steps = 20
            animation_timesteps = np.linspace(0, len(diffusion.betas)-1, animation_steps).astype(int)
            
            frames = []
            for t in animation_timesteps:
                noise = np.random.randn(*data.shape)
                x_t = diffusion.q_sample(data, np.array([t] * len(data)), noise)
                
                frames.append(
                    go.Frame(
                        data=[go.Scatter(x=x_t[:, 0], y=x_t[:, 1], mode='markers',
                                       marker=dict(size=3, color='blue'))],
                        name=str(t)
                    )
                )
            
            # 初始帧
            fig_anim = go.Figure(
                data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers',
                               marker=dict(size=3, color='blue'))],
                layout=go.Layout(
                    title="扩散过程动画" if CHINESE_SUPPORTED else "Diffusion Animation",
                    xaxis=dict(range=[-4, 4], autorange=False),
                    yaxis=dict(range=[-4, 4], autorange=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="播放", method="animate",
                                    args=[None, {"frame": {"duration": 100}}])]
                    )]
                ),
                frames=frames
            )
            
            st.plotly_chart(fig_anim, use_container_width=True)
    
    # Score函数可视化
    with st.expander("📐 Score函数（梯度场）" if CHINESE_SUPPORTED else "📐 Score Function (Gradient Field)"):
        if CHINESE_SUPPORTED:
            st.markdown("""
            Score函数 $\\nabla_x \\log p(x)$ 指向数据密度增加的方向。
            扩散模型本质上是在学习这个score函数。
            """)
        
        # 选择一个时间步
        score_t = st.slider(
            "时间步" if CHINESE_SUPPORTED else "Timestep",
            min_value=0,
            max_value=len(diffusion.betas) - 1,
            value=len(diffusion.betas) // 4,
            key="score_t"
        )
        
        # 生成该时间步的数据
        noise = np.random.randn(*data.shape)
        x_t = diffusion.q_sample(data, np.array([score_t] * len(data)), noise)
        
        # 创建网格
        grid_size = 20
        x_grid = np.linspace(-3, 3, grid_size)
        y_grid = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 简化的score估计（使用KDE）
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x_t.T)
        
        # 计算梯度（数值方法）
        delta = 0.1
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x, y = X[i, j], Y[i, j]
                
                # 数值梯度
                grad_x = (kde([x + delta, y]) - kde([x - delta, y])) / (2 * delta)
                grad_y = (kde([x, y + delta]) - kde([x, y - delta])) / (2 * delta)
                
                U[i, j] = grad_x[0]
                V[i, j] = grad_y[0]
        
        # 绘制
        fig_score = go.Figure()
        
        # 数据点
        fig_score.add_trace(go.Scatter(
            x=x_t[:, 0], y=x_t[:, 1],
            mode='markers',
            marker=dict(size=2, color='lightblue', opacity=0.5),
            name='Data'
        ))
        
        # 梯度场（抽样显示）
        step = 2
        fig_score.add_trace(go.Scatter(
            x=X[::step, ::step].flatten(),
            y=Y[::step, ::step].flatten(),
            mode='markers',
            marker=dict(size=8, color='red', symbol='arrow', angle=np.arctan2(V[::step, ::step], U[::step, ::step]).flatten() * 180 / np.pi),
            name='Score',
            showlegend=False
        ))
        
        fig_score.update_layout(
            title=f"Score函数 at t={score_t}" if CHINESE_SUPPORTED else f"Score Function at t={score_t}",
            xaxis=dict(range=[-3, 3]),
            yaxis=dict(range=[-3, 3]),
            height=500
        )
        
        st.plotly_chart(fig_score, use_container_width=True)
        
        if CHINESE_SUPPORTED:
            st.info("""
            **Score-Based模型观点**：
            - 扩散模型可以看作是在不同噪声水平下学习score函数
            - Score指向密度高的区域，遵循这个场可以从噪声采样到数据
            - 这与Langevin动力学采样相关
            """)
