# 更新日志

## [v2.0.0] - 2024-12-05

### 重大更新：扩散模型模块

#### 新增功能
- ✨ **完整的扩散模型实现** (1403行代码)
  - DDPM (Denoising Diffusion Probabilistic Models) 数学原理
  - DDIM (Denoising Diffusion Implicit Models) 加速采样
  - 前向扩散过程可视化
  - 反向去噪过程演示
  - 图像生成功能
  - 2D数据分布扩散可视化
  - Score函数（梯度场）可视化

#### 核心特性

**📚 数学理论标签页**
- 前向扩散过程的马尔可夫链定义
- 重参数化技巧完整推导
- 反向去噪过程的变分下界（ELBO）推导
- 后验分布贝叶斯推导
- DDPM vs DDIM 算法详细对比
- Beta调度策略可视化（linear, cosine, quadratic）

**➡️ 前向扩散标签页**
- 交互式图像加噪过程
- 支持多种输入图案（圆形、数字、渐变等）
- 不同时间步的噪声可视化
- 信噪比统计分析
- 单步扩散详细过程演示

**⬅️ 反向去噪标签页**
- 理想去噪演示（已知真实噪声）
- 逐步去噪过程可视化
- 支持多种目标图案（圆形、方形、三角形、星形）
- 重建误差计算

**🎨 图像生成标签页**
- DDPM标准采样算法
- DDIM加速采样算法（可调节随机性参数η）
- 采样方法对比工具
- 生成过程逐帧可视化
- 实际应用案例介绍（Stable Diffusion、DALL-E等）

**📊 2D可视化标签页**
- Swiss Roll（瑞士卷）数据分布
- Two Moons（双月）分布
- 同心圆分布
- 高斯混合分布
- 扩散动画生成
- Score函数（梯度场）可视化
- 分布统计分析（均值、标准差、相关系数）

#### 技术实现

**DiffusionModel 类**
```python
- 支持三种Beta调度：linear, cosine, quadratic
- q_sample(): 前向扩散采样
- p_sample(): 反向去噪采样
- 完整的α、ᾱ、后验方差计算
```

**SimpleUNet 类**
```python
- 轻量级U-Net架构
- 时间步位置编码
- 适合教学演示
```

**可视化工具**
- Plotly交互式图表
- 热力图、散点图、向量场
- 动画生成功能

### 文档更新

#### README.md 完全重写
- ✅ 移除夸大宣传，采用务实严谨的描述
- ✅ 明确项目目标和适用人群
- ✅ 详细列出当前限制
- ✅ 制定清晰的未来发展路线图
- ✅ 添加 Just For Dream Lab 开发者信息
- ✅ 强调教学和研究目的

**短期计划（3-6个月）**
- Transformer架构模块
- 优化算法增强
- 正则化技术

**中期计划（6-12个月）**
- 注意力机制深度剖析
- GAN生成对抗网络
- VAE变分自编码器
- 强化学习基础

**长期愿景（1年以上）**
- 大语言模型（LLM）原理
- 多模态模型
- 模型压缩与效率
- 可解释性工具
- 交互式论文复现

#### .gitignore 文件创建
- 完整的Python项目忽略规则
- 虚拟环境、IDE配置
- 模型文件、数据集
- 临时文件和日志
- Streamlit配置
- 跨平台支持（MacOS、Windows、Linux）

### 代码改进

**app.py 修改**
```python
# 新增扩散模型导入
from diffusion import diffusion_tab

# 添加第7个标签页
tab7: 🌊 扩散模型

# 重新组织标签页顺序
1. CNN卷积数学
2. GNN图神经网络
3. RNN/LSTM时序网络
4. 扩散模型 ← 新增
5. 数学推导工具
6. 反向传播原理
7. 交互实验室
```

### Bug修复
- 🐛 修复 multiselect 默认值不在选项列表中的错误
- 🐛 优化时间步选择逻辑，确保边界值正确处理

### 测试验证
- ✅ 所有模块导入成功
- ✅ 核心类定义正确
- ✅ Beta调度算法验证通过
- ✅ Streamlit应用启动正常

### 统计数据
- **新增代码**: 1403行（diffusion.py）
- **文档更新**: README.md 从78行扩展到250+行
- **配置文件**: .gitignore 170行
- **标签页数量**: 从6个增加到7个
- **支持的采样算法**: DDPM, DDIM
- **Beta调度策略**: 3种（linear, cosine, quadratic）
- **2D数据分布**: 4种（Swiss Roll, Two Moons, Concentric Circles, Gaussian Mixture）

### 致谢
开发者：Just For Dream Lab

---

## [v1.0.0] - 之前版本

### 已有功能
- CNN卷积神经网络数学原理
- GNN图神经网络
- RNN/LSTM时序网络
- 数学推导工具
- 反向传播原理
- 交互实验室
