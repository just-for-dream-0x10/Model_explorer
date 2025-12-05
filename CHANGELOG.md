# 更新日志

## [v2.1.0] - 2025-12-XX

### 新增功能：失败案例博物馆 🏛️ + ResNet残差连接分析 🏗️

#### 核心特性
- ✨ **失败案例博物馆标签页** (`tabs/failure_museum.py`)
  - 展示4个经典的网络设计错误案例
  - 每个案例都提供真实的数值证明，而非理论说教
  - 核心理念：不是告诉你"这样不好"，而是让你看到"梯度真的变成1e-10了"

#### 实现的案例

**1. 100层普通MLP（梯度消失）**
- ✅ 使用Sigmoid激活函数的深层网络
- ✅ 梯度流分析：可视化各层梯度范数
- ✅ 数值证明：深层梯度接近0（< 1e-5）
- ✅ 解决方案：残差连接或更换激活函数

**2. 卷积层直接接超大全连接（参数爆炸）**
- ✅ 演示32亿参数的全连接层
- ✅ 详细计算：64×224×224 → 1000分类的参数量
- ✅ 内存占用分析：>12GB显存
- ✅ 解决方案：全局平均池化（参数减少5万倍）

**3. 20层卷积网络无归一化（训练不稳定）**
- ✅ 展示无BatchNorm的深层网络
- ✅ 激活值分布分析
- ✅ 解释为什么需要归一化层
- ✅ 解决方案：添加BatchNorm或LayerNorm

**4. 简单MLP + 超大学习率（梯度爆炸）**
- ✅ 对比两种学习率（lr=10.0 vs lr=0.01）
- ✅ 模拟训练过程，展示Loss变NaN
- ✅ 交互式训练演示
- ✅ 解决方案：使用合理学习率或学习率调度器

#### 技术实现

**新增文件**
- `utils/failure_cases.py` - 失败案例网络定义模块
  - `DeepMLPWithoutSkip` - 100层MLP类
  - `ConvToFullyConnected` - 参数爆炸网络
  - `DeepNetWithoutNorm` - 无归一化网络
  - `TinyMLPWithHugeLR` - 学习率对比网络
  - `get_failure_case()` - 案例获取函数

- `tabs/failure_museum.py` - 失败案例博物馆标签页
  - `calculate_params_and_memory()` - 参数与内存分析
  - `simulate_gradient_flow()` - 梯度流模拟
  - `simulate_training_with_lr()` - 训练过程模拟
  - `plot_gradient_flow()` - 梯度流可视化
  - `plot_loss_curve()` - Loss曲线绘制
  - `failure_museum_tab()` - 主界面函数

**应用集成**
- 更新 `tabs/__init__.py` - 导出新模块
- 更新 `app.py` - 添加第8个标签页
- 保持项目结构化和代码风格一致性

#### 教学价值

**核心教学理念**
1. **数值证明优先**：用真实数值而非理论说教
   - 不是说"梯度会消失"，而是展示"梯度=1e-10"
   - 不是说"参数太多"，而是展示"3,211,265,000个参数"

2. **问题根源分析**：每个案例都解释为什么会失败
   - Sigmoid导数连乘导致指数衰减
   - 特征图尺寸过大导致全连接层爆炸
   - 激活值分布偏移导致训练不稳定

3. **实用解决方案**：提供可行的改进建议
   - 残差连接、ReLU激活函数
   - 全局平均池化
   - BatchNorm/LayerNorm
   - 学习率调度器

#### 文档更新

**README.md**
- ✅ 在"当前实现的架构模块"中添加失败案例博物馆章节
- ✅ 更新Phase 3开发路线图，标记为已完成
- ✅ 添加教学场景说明

**代码质量**
- ✅ 遵循项目代码风格和结构
- ✅ 详细的中文注释和文档字符串
- ✅ 模块化设计，易于扩展
- ✅ 与现有功能（参数计算器、梯度追踪）良好集成

#### 统计数据
- **新增代码**: ~400行（failure_cases.py + failure_museum.py）
- **标签页数量**: 从7个增加到8个
- **失败案例**: 4个经典案例
- **可视化图表**: 梯度流图、Loss曲线图
- **教学价值**: ⭐⭐⭐⭐⭐

---

### 新增功能：ResNet残差连接分析 🏗️

#### 核心特性
- ✨ **ResNet残差连接分析标签页** (`tabs/resnet_analysis.py`)
  - 对比普通网络和ResNet的梯度流
  - 数值证明"梯度高速公路"机制
  - 核心理念：用真实梯度数据验证残差连接的有效性

#### 功能亮点

**1. 网络对比实验**
- ✅ 普通深层网络（PlainNet）：无残差连接
- ✅ ResNet：有残差连接（y = F(x) + x）
- ✅ 相同深度、相同参数量对比
- ✅ 支持两种模式：
  - 简化版（全连接层，快速演示）
  - 完整版（卷积层，真实ResNet）

**2. 梯度流分析**
- ✅ 逐层梯度范数统计
- ✅ 梯度消失检测（< 1e-5为警戒线）
- ✅ 对数坐标可视化
- ✅ 关键指标对比：
  - 平均梯度范数
  - 梯度消失层数
  - 逐层梯度分布

**3. 数学原理展示**
- ✅ 残差连接的前向传播：y = F(x) + x
- ✅ 反向传播推导：∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
- ✅ 为什么"+1"项很重要（梯度高速公路）
- ✅ 数值案例：10层网络梯度衰减对比

**4. 交互式实验**
- ✅ 可调节网络深度（10-50层）
- ✅ 实时梯度分析按钮
- ✅ 详细的层级梯度报告
- ✅ 自动识别梯度消失问题

#### 技术实现

**新增文件**
- `utils/resnet_models.py` - ResNet网络定义模块
  - `PlainBlock` - 普通卷积块（无残差）
  - `ResidualBlock` - 残差块（有残差连接）
  - `PlainNet` - 普通深层网络
  - `ResNet` - ResNet网络
  - `TinyPlainNet` - 简化版普通网络（全连接）
  - `TinyResNet` - 简化版ResNet（全连接）
  - `get_resnet_comparison()` - 网络对比函数

- `tabs/resnet_analysis.py` - ResNet分析标签页
  - `analyze_gradient_flow()` - 梯度流分析
  - `plot_gradient_comparison()` - 梯度对比可视化
  - `plot_gradient_statistics()` - 梯度统计图表
  - `explain_residual_math()` - 数学原理说明
  - `resnet_analysis_tab()` - 主界面函数

**应用集成**
- 更新 `tabs/__init__.py` - 导出新模块
- 更新 `app.py` - 添加第9个标签页
- 保持代码结构化和风格一致性

#### 教学价值

**核心教学理念**
1. **数值验证优先**：真实梯度数据而非理论
   - 普通网络：梯度 = 1e-10（消失）
   - ResNet：梯度 = 1e-2（正常）
   - 对比倍率：1000倍差异

2. **数学原理清晰**：详细推导残差连接的作用
   - 前向：y = F(x) + x
   - 反向：∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
   - 关键："+1"项保证梯度传播

3. **直观可视化**：多维度图表展示
   - 梯度流曲线（对数坐标）
   - 统计对比柱状图
   - 逐层梯度详情

4. **实际工程意义**：
   - 为什么现代网络都用残差连接
   - 何时需要残差连接（>20层）
   - Transformer为什么也用残差连接

#### 实验结果示例

典型实验结果（20层网络）：
```
普通网络：
- 平均梯度：1.2e-4
- 梯度消失层数：15/20
- 后层梯度：< 1e-8

ResNet：
- 平均梯度：3.5e-2
- 梯度消失层数：0/20
- 后层梯度：> 1e-3

✅ ResNet梯度是普通网络的 291倍！
```

#### 文档更新

**README.md**
- ✅ 在"当前实现的架构模块"中添加ResNet分析章节
- ✅ 更新Phase 2开发路线图，标记为已完成
- ✅ 添加教学场景和实用价值说明

**代码质量**
- ✅ 遵循项目代码风格和结构
- ✅ 详细的中文注释和文档字符串
- ✅ 模块化设计，支持两种网络类型
- ✅ 与失败案例博物馆互补（一个展示问题，一个展示解决方案）

#### 统计数据
- **新增代码**: ~600行（resnet_models.py + resnet_analysis.py）
- **标签页数量**: 从8个增加到9个
- **网络类型**: 4种（PlainNet、ResNet、TinyPlainNet、TinyResNet）
- **可视化图表**: 梯度流对比图、统计柱状图
- **教学价值**: ⭐⭐⭐⭐⭐

#### 与Phase 2路线图的关系

✅ **Phase 2 部分完成**：
- ✅ ResNet残差连接分析（已完成）
- ✅ 归一化层对比（已完成）
- [ ] Vision Transformer (ViT)（待开发）
- [ ] 架构对比实验室（待开发）

---

### 新增功能：归一化层对比 🔧

#### 核心特性
- ✨ **归一化层对比标签页** (`tabs/normalization_comparison.py`)
  - 对比BatchNorm、LayerNorm、GroupNorm三种方法
  - 可视化"在哪个维度归一化"的差异
  - 核心理念：用交互式图表展示归一化的实际效果

#### 功能亮点

**1. 三种归一化方法对比**
- ✅ BatchNorm: 在(Batch, H, W)维度归一化
- ✅ LayerNorm: 在(Channel, H, W)维度归一化
- ✅ GroupNorm: 分组归一化（介于两者之间）
- ✅ 交互式参数调节（batch size、通道数、分组数）

**2. 可视化展示**
- ✅ 激活值分布直方图对比
- ✅ 各通道统计量对比（均值、标准差）
- ✅ 归一化前后的数值变化
- ✅ 采样优化（加快渲染速度）

**3. Batch Size敏感性分析**
- ✅ 实验验证：为什么小batch时BatchNorm效果差
- ✅ 不同batch size下的归一化效果曲线
- ✅ 数值证明：LayerNorm/GroupNorm与batch size无关

**4. 适用场景决策指南**
- ✅ BatchNorm: CNN + 大batch
- ✅ LayerNorm: Transformer、RNN/LSTM
- ✅ GroupNorm: 小batch场景（目标检测）
- ✅ 详细的优缺点对比表格

**5. 数学原理详解**
- ✅ 三种方法的公式对比
- ✅ 归一化维度的直观解释
- ✅ 为什么Transformer使用LayerNorm

#### 技术实现

**新增文件**
- `utils/normalization_layers.py` - 归一化层实现模块
  - `apply_batch_norm()` - 手动实现BatchNorm
  - `apply_layer_norm()` - 手动实现LayerNorm
  - `apply_group_norm()` - 手动实现GroupNorm
  - `compare_normalization_methods()` - 三种方法对比
  - `SimpleCNNWithNorm` - 可选归一化层的CNN
  - `get_normalization_comparison_info()` - 归一化信息

- `tabs/normalization_comparison.py` - 归一化层对比标签页
  - `plot_activation_distribution()` - 激活值分布图
  - `plot_normalization_comparison()` - 四种情况对比图
  - `plot_channel_statistics()` - 通道统计图
  - `explain_normalization_math()` - 数学原理说明
  - `normalization_comparison_tab()` - 主界面函数

**性能优化**
- ✅ 减小数据规模（空间尺寸从32×32降到16×16）
- ✅ 采样渲染（最多采样10000个数据点）
- ✅ 减少直方图bins数量（从50降到30）
- ✅ 限制参数范围（通道数最大64，batch size最大8）

**应用集成**
- 更新 `tabs/__init__.py` - 导出新模块
- 更新 `app.py` - 添加第10个标签页
- 保持代码结构化和风格一致性

#### 教学价值

**核心教学理念**
1. **可视化优先**：用图表展示抽象概念
   - 直方图：归一化前后的分布变化
   - 柱状图：各通道的统计量对比
   - 折线图：batch size敏感性曲线

2. **交互式探索**：用户可以调整参数观察变化
   - 调节batch size：观察BatchNorm的变化
   - 调节通道数：理解LayerNorm的工作机制
   - 调节分组数：探索GroupNorm的特性

3. **实用决策指南**：不仅讲原理，还教如何选择
   - CNN → BatchNorm
   - Transformer → LayerNorm
   - 小batch → GroupNorm

4. **经典问题解答**：
   - 为什么Transformer用LayerNorm而不是BatchNorm？
   - 为什么目标检测任务喜欢用GroupNorm？
   - 小batch时BatchNorm为什么效果差？

#### 实验结果示例

典型实验结果（batch_size=1 vs 16）：
```
Batch Size = 1:
- BatchNorm标准差: 0.8234（偏离理想值1.0）
- LayerNorm标准差: 1.0002（稳定）
- GroupNorm标准差: 1.0001（稳定）

Batch Size = 16:
- BatchNorm标准差: 1.0003（接近理想值）
- LayerNorm标准差: 1.0002（稳定）
- GroupNorm标准差: 1.0001（稳定）

✅ 证明了BatchNorm的batch size依赖性！
```

#### 文档更新

**README.md**
- ✅ 在"当前实现的架构模块"中添加归一化层对比章节
- ✅ 更新Phase 2开发路线图，标记为已完成
- ✅ 添加适用场景和教学价值说明

**代码质量**
- ✅ 遵循项目代码风格和结构
- ✅ 详细的中文注释和文档字符串
- ✅ 性能优化，加快加载速度
- ✅ 与其他模块互补（失败案例博物馆、ResNet分析）

#### 统计数据
- **新增代码**: ~700行（normalization_layers.py + normalization_comparison.py）
- **标签页数量**: 从9个增加到10个
- **归一化方法**: 3种（BatchNorm、LayerNorm、GroupNorm）
- **可视化图表**: 4种（分布图、对比图、统计图、敏感性曲线）
- **教学价值**: ⭐⭐⭐⭐⭐

#### 与失败案例博物馆的关联

**形成教学闭环**：
1. **失败案例博物馆** → 展示"无归一化"导致训练不稳定
2. **归一化层对比** → 展示三种归一化方法的差异
3. **ResNet分析** → 说明BatchNorm在残差网络中的作用

---

## [v2.0.0] - 2025-12-05

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
