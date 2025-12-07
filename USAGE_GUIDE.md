# 使用指南

本文档介绍重构后项目的各个模块的使用方法和最佳实践。

## 📚 目录

- [快速开始](#快速开始)
- [核心模块使用](#核心模块使用)
- [可视化工具](#可视化工具)
- [缓存系统](#缓存系统)
- [性能监控](#性能监控)
- [开发指南](#开发指南)

## 🚀 快速开始

### 基本导入

```python
# 核心模块
from utils import ChartBuilder, NetworkAnalyzer, MemoryAnalyzer
from utils.visualization import PlotHelper, NetworkVisualization
from utils.cache import cached, get_cache_manager
from utils.performance_monitor import get_performance_monitor, monitor_operation
```

### 快速示例

```python
# 创建图表
chart = ChartBuilder()
fig = chart.create_line_chart([1,2,3], [1,4,9], "平方函数")
chart.display_chart(fig)

# 分析网络
analyzer = NetworkAnalyzer((3, 224, 224))
analyzer.add_layer({
    'layer_type': 'conv2d',
    'params': {
        'in_channels': 3,
        'out_channels': 64,
        'kernel_size': 7,
        'stride': 2,
        'padding': 3
    }
})
result = analyzer.analyze_network()
```

## 🎯 核心模块使用

### NetworkAnalyzer - 网络分析器

用于分析神经网络架构的参数量、FLOPs和内存占用。

```python
from utils.core import NetworkAnalyzer

# 创建分析器
analyzer = NetworkAnalyzer(input_shape=(3, 224, 224))

# 添加层配置
analyzer.add_layer({
    'layer_type': 'conv2d',
    'params': {
        'in_channels': 3,
        'out_channels': 64,
        'kernel_size': 7,
        'stride': 2,
        'padding': 3,
        'use_bias': True
    }
})

# 分析网络
result = analyzer.analyze_network()

# 查看结果
print(f"总参数量: {result['total_parameters']:,}")
print(f"总FLOPs: {result['total_flops']:,}")
print(f"峰值内存: {result['peak_memory_mb']:.2f}MB")
```

### ParameterCalculator - 参数计算器

提供各种网络层的详细计算功能。

```python
from utils.core import ParameterCalculator

calc = ParameterCalculator()

# 分析卷积层
conv_result = calc.calculate_conv2d_params(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    input_shape=(3, 224, 224)
)

print(f"参数量: {conv_result['parameters']['total']:,}")
print(f"FLOPs: {conv_result['flops']['flops_readable']}")
print(f"内存: {conv_result['param_memory_mb']:.2f}MB")
```

### MemoryAnalyzer - 内存分析器

分析网络内存占用和可行性。

```python
from utils.core import MemoryAnalyzer

analyzer = MemoryAnalyzer(dtype='float32')

# 分析单层内存
memory_result = analyzer.analyze_layer_memory(
    layer_type='conv2d',
    params={'in_channels': 3, 'out_channels': 64, 'kernel_size': 7},
    input_shape=(3, 224, 224)
)

# 分析整个网络
layers = [
    {'layer_type': 'conv2d', 'params': {...}},
    {'layer_type': 'linear', 'params': {...}}
]
network_result = analyzer.analyze_network_memory(
    layers=layers,
    input_shape=(3, 224, 224),
    batch_size=32
)
```

## 🎨 可视化工具

### ChartBuilder - 统一图表工具

提供统一的图表创建接口，支持多种图表类型。

```python
from utils.visualization import ChartBuilder

chart = ChartBuilder()

# 折线图
fig = chart.create_line_chart(
    x_data=[1, 2, 3, 4],
    y_data=[[1, 4, 9, 16], [1, 2, 3, 4]],
    title="函数对比",
    line_names=["平方", "线性"],
    height=400
)

# 柱状图
fig_bar = chart.create_bar_chart(
    x_data=["A", "B", "C"],
    y_data=[10, 20, 30],
    title="数据对比"
)

# 热力图
fig_heatmap = chart.create_heatmap(
    data=np.random.rand(10, 10),
    title="热力图",
    colorscale="Viridis"
)

# 显示图表
chart.display_chart(fig)
```

### PlotHelper - 辅助工具

提供常用的辅助功能。

```python
from utils.visualization import PlotHelper

# 格式化数字
formatted = PlotHelper.format_number(1234567)  # "1.23M"
formatted = PlotHelper.format_number(0.00123)  # "1.23m"

# 创建对比表格
data = {
    "模型A": {"参数量": "10M", "准确率": "95%"},
    "模型B": {"参数量": "5M", "准确率": "93%"}
}
PlotHelper.create_comparison_table(data, "模型对比")

# 显示指标卡片
metrics = {"准确率": "95%", "损失": "0.05"}
PlotHelper.show_metrics(metrics, columns=2)
```

### NetworkVisualization - 网络可视化

专门用于神经网络的可视化。

```python
from utils.visualization import NetworkVisualization

viz = NetworkVisualization()

# 绘制网络架构
fig = viz.plot_network_architecture(
    layer_shapes=[(3, 224, 224), (64, 112, 112), (128, 56, 56)],
    layer_names=["Input", "Conv1", "Conv2"],
    title="网络架构图"
)

# 绘制训练曲线
curves_data = {
    "Model A": {
        'epochs': [1, 2, 3, 4, 5],
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.85, 0.65, 0.5, 0.4, 0.35]
    }
}
fig = viz.plot_training_curves(curves_data, metric="loss")
```

### MathVisualization - 数学可视化

用于数学概念的可视化展示。

```python
from utils.visualization import MathVisualization

math_viz = MathVisualization()

# 绘制卷积过程
input_matrix = np.random.rand(5, 5)
kernel = np.random.rand(3, 3)
output_matrix = np.random.rand(3, 3)

fig = math_viz.plot_convolution_process(
    input_matrix=input_matrix,
    kernel=kernel,
    output_matrix=output_matrix,
    title="卷积过程可视化"
)

# 绘制激活函数
fig = math_viz.plot_activation_functions(x_range=(-5, 5))
```

## 🗄️ 缓存系统

### 基本缓存使用

```python
from utils import cached

@cached(ttl=3600)  # 缓存1小时
def expensive_calculation(x: int) -> int:
    """耗时的计算"""
    time.sleep(0.1)  # 模拟耗时操作
    return sum(range(x))

# 第一次调用（会执行计算）
result1 = expensive_calculation(1000)

# 第二次调用（从缓存获取）
result2 = expensive_calculation(1000)
```

### 缓存管理器

```python
from utils.cache import CacheManager

# 创建缓存管理器
cache = CacheManager(max_size=1000, default_ttl=3600)

# 手动缓存操作
cache.set("key1", "value1", ttl=1800)
value = cache.get("key1")
cache.delete("key1")

# 获取缓存统计
stats = cache.get_stats()
print(f"总条目: {stats['total_entries']}")
print(f"使用率: {stats['usage_ratio']:.1%}")
```

### 缓存装饰器

```python
from utils import cached_method

class MyClass:
    @cached_method(ttl=1800)
    def slow_method(self, param1, param2):
        # 缓存实例方法
        return self._expensive_computation(param1, param2)
```

### 自定义缓存键

```python
from utils import cached, network_analysis_key

@cached(key_func=network_analysis_key)
def analyze_network(input_shape, layers):
    # 使用自定义键的缓存
    return complex_analysis(input_shape, layers)
```

## 📊 性能监控

### 基本监控

```python
from utils.performance_monitor import get_performance_monitor, monitor_operation

# 获取监控器
monitor = get_performance_monitor()

# 开始监控
monitor.start_monitoring(interval=1.0)

# 监控操作
with monitor_operation("数据处理"):
    # 你的代码
    process_data()

# 停止监控
monitor.stop_monitoring()
```

### 性能仪表板

```python
# 在Streamlit应用中显示
from utils.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
if st.button("显示性能仪表板"):
    monitor.display_performance_dashboard()
```

### 操作计时

```python
from utils.performance_monitor import monitor_operation

# 监控特定操作
with monitor_operation("模型训练"):
    train_model()

# 查看操作统计
stats = monitor.get_operation_stats("模型训练")
print(f"平均耗时: {stats['avg_time']:.3f}s")
```

## 🛠️ 开发指南

### 添加新的图表类型

```python
# 在ChartBuilder中添加新方法
class ChartBuilder:
    def create_new_chart_type(self, data, **kwargs):
        """新图表类型"""
        fig = go.Figure()
        # 实现图表逻辑
        return fig
```

### 添加新的分析器

```python
# 在core目录下创建新模块
# utils/core/new_analyzer.py

class NewAnalyzer:
    @staticmethod
    def analyze_layer(params):
        """分析新层类型"""
        return analysis_result
```

### 异常处理最佳实践

```python
from utils.exceptions import NetworkAnalysisError, ComputationError

def safe_analysis(params):
    try:
        return analyze_network(params)
    except InvalidLayerConfigError as e:
        logger.error(f"层配置错误: {e}")
        return None
    except ComputationError as e:
        logger.error(f"计算错误: {e}")
        raise
```

### 测试新功能

```python
# tests/test_new_feature.py
import pytest
from utils.new_module import NewFeature

def test_new_feature():
    feature = NewFeature()
    result = feature.process()
    assert result is not None
```

## 📈 性能优化建议

### 1. 使用缓存

```python
# ✅ 好的做法
@cached(ttl=3600)
def expensive_operation(params):
    return complex_calculation(params)

# ❌ 避免的做法
def expensive_operation(params):
    return complex_calculation(params)  # 每次都重新计算
```

### 2. 批量处理

```python
# ✅ 好的做法
def batch_process(data_list):
    results = []
    for batch in chunked(data_list, 32):
        results.append(process_batch(batch))
    return results

# ❌ 避免的做法
def process_individually(data_list):
    return [process_item(item) for item in data_list]
```

### 3. 内存管理

```python
# ✅ 好的做法
def memory_efficient_function():
    result = []
    for item in large_iterator:
        # 处理后立即释放
        processed = process_item(item)
        result.append(processed)
    return result

# ❌ 避免的做法
def memory_inefficient_function():
    all_items = list(large_iterator)  # 一次性加载所有数据
    return [process_item(item) for item in all_items]
```

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```python
   # 检查模块路径和依赖
   from utils.visualization import ChartBuilder  # 确保路径正确
   ```

2. **缓存不工作**
   ```python
   # 检查缓存管理器状态
   cache = get_cache_manager()
   stats = cache.get_stats()
   print(stats)
   ```

3. **性能监控数据异常**
   ```python
   # 重启监控器
   monitor.stop_monitoring()
   monitor.start_monitoring()
   ```

## 📞 更多资源

- [API参考文档](./API_REFERENCE.md)
- [架构设计文档](./ARCHITECTURE.md)
- [贡献指南](./CONTRIBUTING.md)

---

**更新时间**: 2025年12月6日  
**版本**: 2.0.0