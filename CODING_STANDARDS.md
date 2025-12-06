# 编码规范文档

## 🎯 目标

建立统一的编码标准，提高代码质量、可读性和可维护性。

## 📋 目录

- [代码风格](#代码风格)
- [命名规范](#命名规范)
- [模块组织](#模块组织)
- [文档规范](#文档规范)
- [错误处理](#错误处理)
- [性能优化](#性能优化)
- [测试规范](#测试规范)
- [代码审查](#代码审查)

## 🎨 代码风格

### Python代码规范

遵循 **PEP 8** 标准，并补充以下规定：

#### 1. 行长度与缩进
```python
# ✅ 正确：使用4个空格缩进，行长度不超过88字符
def calculate_network_params(
    input_shape: Tuple[int, ...],
    layers: List[LayerConfig]
) -> Dict[str, Any]:
    """计算网络参数量和内存占用"""
    pass

# ❌ 错误：行长度过长
def calculate_network_params(input_shape: Tuple[int, int, int], layers: List[LayerConfig]) -> Dict[str, Any]:
    pass
```

#### 2. 导入语句
```python
# ✅ 正确：按标准库、第三方库、本地模块分组
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.config import CHINESE_SUPPORTED
from utils.exceptions import NetworkAnalysisError

# ❌ 错误：混合导入顺序
import streamlit as st
import os
from utils.config import CHINESE_SUPPORTED
import numpy as np
```

#### 3. 字符串格式化
```python
# ✅ 正确：使用f-string
name = "CNN"
params = 1000
message = f"{name}网络有{params}个参数"

# ❌ 错误：使用%格式化
message = "%s网络有%d个参数" % (name, params)
```

## 🏷️ 命名规范

### 1. 变量和函数
```python
# ✅ 正确：使用snake_case
input_shape = (224, 224, 3)
learning_rate = 0.001
def calculate_flops():
    pass

# ❌ 错误：使用camelCase
inputShape = (224, 224, 3)
learningRate = 0.001
def calculateFlops():
    pass
```

### 2. 类名
```python
# ✅ 正确：使用PascalCase
class NetworkAnalyzer:
    pass

class ConvolutionLayer:
    pass

# ❌ 错误：使用snake_case
class network_analyzer:
    pass
```

### 3. 常量
```python
# ✅ 正确：使用UPPER_CASE
DEFAULT_BATCH_SIZE = 32
MAX_IMAGE_SIZE = 1024
SUPPORTED_ACTIVATIONS = ["relu", "sigmoid", "tanh"]

# ❌ 错误：使用小写
default_batch_size = 32
```

### 4. 私有成员
```python
# ✅ 正确：使用单下划线前缀
class NetworkAnalyzer:
    def _calculate_internal_params(self):
        """内部计算方法"""
        pass
    
    def __private_method(self):
        """私有方法"""
        pass

# ❌ 错误：不使用下划线
class NetworkAnalyzer:
    def calculate_internal_params(self):
        pass
```

## 📁 模块组织

### 1. 文件结构
```
project/
├── app.py                    # 主入口文件
├── cnn.py                    # 核心CNN模块
├── rnn_lstm.py              # 核心RNN模块
├── gnn.py                    # 核心GNN模块
├── tabs/                     # 功能标签页
│   ├── __init__.py
│   ├── params_calculator.py
│   ├── architecture_comparison.py
│   └── ...
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   └── visualization/
│       ├── __init__.py
│       ├── chart_utils.py
│       └── plot_helpers.py
└── templates/                # 模板配置
    ├── configs/
    └── template_loader.py
```

### 2. 模块内容顺序
```python
# 1. 模块文档字符串
"""模块功能描述"""

# 2. 导入语句
import os
from typing import Dict, List

import numpy as np

# 3. 常量定义
DEFAULT_BATCH_SIZE = 32

# 4. 异常类定义
class CustomError(Exception):
    """自定义异常"""
    pass

# 5. 工具函数
def helper_function():
    """辅助函数"""
    pass

# 6. 类定义
class MainClass:
    """主要类"""
    pass

# 7. 主函数（如果适用）
def main():
    """主函数"""
    pass
```

## 📖 文档规范

### 1. 模块文档
```python
"""CNN卷积神经网络数学原理模块

本模块提供CNN卷积操作的详细数学分析和可视化功能，包括：
- 卷积操作的逐像素计算演示
- 参数量、FLOPs和内存占用分析
- 不同卷积核类型的对比实验

Author: Just For Dream Lab
Version: 1.0.0
"""
```

### 2. 类文档
```python
class NetworkAnalyzer:
    """网络分析器
    
    用于分析神经网络的参数量、计算复杂度和内存占用。
    
    Attributes:
        input_shape (Tuple[int, ...]): 输入张量形状
        layers (List[LayerConfig]): 网络层配置列表
        
    Example:
        >>> analyzer = NetworkAnalyzer((1, 224, 224))
        >>> analyzer.add_conv_layer(3, 64, 3)
        >>> params = analyzer.calculate_params()
    """
```

### 3. 函数文档
```python
def calculate_conv_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    input_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """计算卷积层的参数量和内存占用
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        input_shape: 输入形状 (C, H, W)
        
    Returns:
        包含以下键的字典：
        - 'parameters': 参数量
        - 'memory_mb': 内存占用(MB)
        - 'flops': 浮点运算次数
        
    Raises:
        ValueError: 当输入参数无效时
        
    Example:
        >>> result = calculate_conv_params(3, 64, 3, (3, 224, 224))
        >>> print(f"参数量: {result['parameters']}")
    """
```

## ⚠️ 错误处理

### 1. 异常定义
```python
# utils/exceptions.py
class NetworkAnalysisError(Exception):
    """网络分析相关异常基类"""
    pass

class InvalidLayerConfigError(NetworkAnalysisError):
    """无效层配置异常"""
    pass

class InsufficientMemoryError(NetworkAnalysisError):
    """内存不足异常"""
    pass
```

### 2. 异常处理模式
```python
# ✅ 正确：具体异常处理
def analyze_network(network_config):
    try:
        result = perform_analysis(network_config)
    except InvalidLayerConfigError as e:
        logger.error(f"层配置无效: {e}")
        return None
    except InsufficientMemoryError as e:
        logger.warning(f"内存不足: {e}")
        return optimize_memory_usage(network_config)
    except Exception as e:
        logger.error(f"未知错误: {e}")
        raise NetworkAnalysisError(f"网络分析失败: {e}")
    
    return result

# ❌ 错误：过于宽泛的异常处理
def analyze_network(network_config):
    try:
        result = perform_analysis(network_config)
    except:
        return None  # 静默失败
    return result

# ❌ 错误：裸露的except
def analyze_network(network_config):
    try:
        result = perform_analysis(network_config)
    except:  # 避免裸露的except
        pass
```

### 3. 错误信息规范
```python
# ✅ 正确：提供详细错误信息
if not isinstance(kernel_size, int) or kernel_size <= 0:
    raise ValueError(
        f"卷积核大小必须为正整数，当前值: {kernel_size} (类型: {type(kernel_size)})"
    )

# ❌ 错误：错误信息不明确
if not isinstance(kernel_size, int) or kernel_size <= 0:
    raise ValueError("参数错误")
```

## ⚡ 性能优化

### 1. 缓存策略
```python
# ✅ 正确：使用LRU缓存
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_im2col_indices(
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    padding: int
) -> Tuple[np.ndarray, np.ndarray]:
    """计算im2col索引，使用缓存避免重复计算"""
    pass

# ✅ 正确：手动缓存复杂计算结果
class NetworkAnalyzer:
    def __init__(self):
        self._param_cache = {}
        self._memory_cache = {}
    
    def get_params(self, layer_id: str) -> Dict[str, Any]:
        if layer_id not in self._param_cache:
            self._param_cache[layer_id] = self._calculate_params(layer_id)
        return self._param_cache[layer_id]
```

### 2. 内存优化
```python
# ✅ 正确：使用生成器减少内存占用
def generate_layer_configs(network_config):
    """生成器方式产生层配置，减少内存占用"""
    for layer in network_config['layers']:
        yield LayerConfig.from_dict(layer)

# ✅ 正确：及时释放大数组
def process_large_matrix(matrix: np.ndarray):
    """处理大矩阵，及时释放内存"""
    result = expensive_computation(matrix)
    del matrix  # 及时释放
    return result

# ❌ 错误：创建不必要的中间数组
def inefficient_computation(data):
    # 创建多个中间数组，占用大量内存
    temp1 = data.copy()
    temp2 = temp1 * 2
    temp3 = temp2 + 1
    return temp3.sum()
```

### 3. 向量化操作
```python
# ✅ 正确：使用numpy向量化
def vectorized_convolution(image, kernel):
    """使用numpy向量化操作"""
    return signal.convolve2d(image, kernel, mode='same')

# ❌ 错误：使用Python循环
def slow_convolution(image, kernel):
    """使用Python循环，性能差"""
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 手动计算卷积
            pass
    return result
```

## 🧪 测试规范

### 1. 测试文件结构
```
tests/
├── __init__.py
├── test_chart_utils.py
├── test_network_analyzer.py
├── test_param_calculator.py
└── conftest.py              # pytest配置
```

### 2. 测试用例规范
```python
# ✅ 正确：完整的测试用例
import pytest
from utils.chart_utils import ChartBuilder

class TestChartBuilder:
    """ChartBuilder测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.sample_data = [1, 2, 3, 4, 5]
    
    def test_create_line_chart_valid_data(self):
        """测试有效数据创建折线图"""
        fig = ChartBuilder.create_line_chart(
            self.sample_data, 
            "测试图表"
        )
        
        assert fig is not None
        assert fig.layout.title.text == "测试图表"
        assert len(fig.data) == 1
    
    def test_create_line_chart_empty_data(self):
        """测试空数据抛出异常"""
        with pytest.raises(ValueError, match="数据不能为空"):
            ChartBuilder.create_line_chart([], "空数据图表")
    
    @pytest.mark.parametrize("data_size", [10, 100, 1000])
    def test_create_line_chart_different_sizes(self, data_size):
        """参数化测试不同数据大小"""
        data = list(range(data_size))
        fig = ChartBuilder.create_line_chart(data, f"大小{data_size}")
        assert len(fig.data[0].x) == data_size
```

### 3. 测试覆盖率要求
- **单元测试覆盖率**: ≥ 80%
- **集成测试覆盖率**: ≥ 60%
- **关键路径覆盖率**: 100%

## 🔍 代码审查

### 1. 审查清单

#### 功能性
- [ ] 代码实现了需求规格
- [ ] 边界条件处理正确
- [ ] 错误处理完善
- [ ] 性能满足要求

#### 代码质量
- [ ] 遵循编码规范
- [ ] 代码可读性好
- [ ] 注释完整准确
- [ ] 没有明显的代码异味

#### 测试
- [ ] 测试覆盖率达标
- [ ] 测试用例有意义
- [ ] 边界条件有测试
- [ ] 异常情况有测试

### 2. 审查流程
1. **自检** - 提交前自行检查
2. **同行审查** - 至少一人审查
3. **自动检查** - CI/CD自动运行
4. **问题修复** - 及时修复发现的问题
5. **再次审查** - 重大修改需要再次审查

## 📝 代码示例

### 完整的模块示例
```python
"""网络参数计算工具

提供神经网络参数量、FLOPs和内存占用的精确计算功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any
from functools import lru_cache
import numpy as np

from utils.exceptions import InvalidLayerConfigError


class NetworkCalculator:
    """网络计算器
    
    用于计算神经网络的各种性能指标，包括参数量、FLOPs和内存占用。
    
    Attributes:
        layers: 网络层配置列表
        input_shape: 输入张量形状
        
    Example:
        >>> calc = NetworkCalculator((1, 224, 224))
        >>> calc.add_conv_layer(3, 64, 3)
        >>> result = calc.calculate_all()
        >>> print(f"总参数量: {result['total_params']}")
    """
    
    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        """初始化计算器
        
        Args:
            input_shape: 输入张量形状
            
        Raises:
            ValueError: 当输入形状无效时
        """
        if not input_shape or len(input_shape) < 3:
            raise ValueError(f"输入形状无效: {input_shape}")
            
        self.input_shape = input_shape
        self.layers: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}
    
    def add_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> None:
        """添加卷积层配置
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            
        Raises:
            InvalidLayerConfigError: 当层配置无效时
        """
        if any(x <= 0 for x in [in_channels, out_channels, kernel_size]):
            raise InvalidLayerConfigError(
                f"卷积层参数必须为正数: "
                f"in_channels={in_channels}, out_channels={out_channels}, "
                f"kernel_size={kernel_size}"
            )
        
        layer_config = {
            'type': 'conv2d',
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        
        self.layers.append(layer_config)
        self._cache.clear()  # 清除缓存
    
    @lru_cache(maxsize=128)
    def _calculate_conv_params(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, int]:
        """计算卷积层参数（缓存版本）
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            input_shape: 输入形状
            
        Returns:
            参数统计字典
        """
        C_in, H_in, W_in = input_shape
        
        # 计算输出尺寸
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        
        # 计算参数量
        weight_params = out_channels * in_channels * kernel_size * kernel_size
        bias_params = out_channels  # 假设使用偏置
        
        # 计算FLOPs
        flops = 2 * weight_params * H_out * W_out  # 乘加操作
        
        return {
            'weight_params': weight_params,
            'bias_params': bias_params,
            'total_params': weight_params + bias_params,
            'flops': flops,
            'output_shape': (out_channels, H_out, W_out)
        }
    
    def calculate_all(self) -> Dict[str, Any]:
        """计算网络整体指标
        
        Returns:
            包含所有计算结果的字典
            
        Raises:
            InvalidLayerConfigError: 当网络配置无效时
        """
        if not self.layers:
            raise InvalidLayerConfigError("网络中没有配置任何层")
        
        total_params = 0
        total_flops = 0
        current_shape = self.input_shape
        
        layer_results = []
        
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'conv2d':
                result = self._calculate_conv_params(
                    layer['in_channels'],
                    layer['out_channels'],
                    layer['kernel_size'],
                    layer['stride'],
                    layer['padding'],
                    current_shape
                )
                
                total_params += result['total_params']
                total_flops += result['flops']
                current_shape = result['output_shape']
                
                layer_results.append({
                    'layer_index': i,
                    'layer_type': 'conv2d',
                    **result
                })
        
        return {
            'total_params': total_params,
            'total_flops': total_flops,
            'final_shape': current_shape,
            'layer_details': layer_results,
            'param_memory_mb': total_params * 4 / (1024**2),  # FP32
            'summary': self._generate_summary(total_params, total_flops)
        }
    
    def _generate_summary(self, total_params: int, total_flops: int) -> str:
        """生成计算结果摘要
        
        Args:
            total_params: 总参数量
            total_flops: 总FLOPs
            
        Returns:
            摘要字符串
        """
        params_readable = (
            f"{total_params / 1e6:.2f}M"
            if total_params > 1e6
            else f"{total_params / 1e3:.2f}K"
        )
        
        flops_readable = (
            f"{total_flops / 1e9:.2f}G"
            if total_flops > 1e9
            else f"{total_flops / 1e6:.2f}M"
        )
        
        return (
            f"网络总参数量: {params_readable} "
            f"({total_params:,})\n"
            f"总计算量: {flops_readable} FLOPs "
            f"({total_flops:,})"
        )


def main() -> None:
    """主函数：演示网络计算器使用"""
    # 创建计算器
    calculator = NetworkCalculator((3, 224, 224))
    
    # 添加层
    calculator.add_conv_layer(3, 64, 7, stride=2, padding=3)
    calculator.add_conv_layer(64, 128, 3, stride=2, padding=1)
    calculator.add_conv_layer(128, 256, 3, stride=2, padding=1)
    
    # 计算结果
    try:
        result = calculator.calculate_all()
        print(result['summary'])
    except InvalidLayerConfigError as e:
        print(f"配置错误: {e}")


if __name__ == "__main__":
    main()
```

## 📚 参考资料

- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350884)
- [The Pragmatic Programmer](https://www.amazon.com/Pragmatic-Programmer-journey-mastery-Anniversary/dp/0135957052)

---

**最后更新**: 2025年12月6日  
**维护者**: Just For Dream Lab  
**版本**: 1.0.0