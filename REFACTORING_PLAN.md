# 代码重构计划

## 📊 现状分析

通过代码分析发现的主要问题：

### 🚨 高优先级问题

1. **Plotly图表代码重复严重**
   - 77处 `st.plotly_chart` 调用
   - 61处 `fig.update_layout` 调用  
   - 41处 `fig = go.Figure` 创建
   - 大量重复的图表配置代码

2. **异常处理不规范**
   - 25处异常处理，部分过于宽泛
   - 存在裸露的 `except:` 语句
   - 错误信息不够详细

3. **模块耦合度高**
   - utils包功能混杂
   - 标签页代码过长（部分超过2000行）
   - 缺乏统一的接口规范

### 🔧 中优先级问题

1. **性能优化空间**
   - 缺乏计算缓存机制
   - 频繁的numpy数组创建
   - 图表重复渲染

2. **代码复用性差**
   - 相似功能重复实现
   - 缺乏公共工具函数
   - 硬编码数值较多

## 🎯 重构目标

1. **提升代码质量** - 减少重复，提高可维护性
2. **改善性能** - 添加缓存，优化计算
3. **增强可测试性** - 模块解耦，便于单元测试
4. **统一代码风格** - 建立编码规范

## 📋 重构计划

### Phase 1: 图表组件重构 (1-2天)

#### 1.1 创建通用图表工具类
```python
# utils/chart_utils.py
class ChartBuilder:
    @staticmethod
    def create_line_chart(data, title, **kwargs):
        # 统一的折线图创建逻辑
        
    @staticmethod  
    def create_bar_chart(data, title, **kwargs):
        # 统一的柱状图创建逻辑
        
    @staticmethod
    def create_heatmap(data, title, **kwargs):
        # 统一的热力图创建逻辑
```

#### 1.2 重构现有图表调用
- 替换77处 `st.plotly_chart` 调用
- 统一图表样式和配置
- 减少重复代码60%+

### Phase 2: 异常处理规范化 (1天)

#### 2.1 创建自定义异常类
```python
# utils/exceptions.py
class NetworkAnalysisError(Exception):
    """网络分析相关异常"""
    
class VisualizationError(Exception):
    """可视化相关异常"""
```

#### 2.2 重构异常处理
- 替换裸露的 `except:` 语句
- 添加具体的异常类型
- 提供详细的错误信息

### Phase 3: 模块解耦重构 (2-3天)

#### 3.1 重构utils包
```
utils/
├── core/           # 核心计算逻辑
│   ├── network_analyzer.py
│   ├── param_calculator.py
│   └── memory_analyzer.py
├── visualization/  # 可视化工具
│   ├── chart_utils.py
│   └── plot_helpers.py
├── models/         # 模型定义
│   ├── network_templates.py
│   └── failure_cases.py
└── config/         # 配置管理
    ├── settings.py
    └── constants.py
```

#### 3.2 重构长文件
- 拆分超过2000行的标签页文件
- 提取公共逻辑到独立模块
- 建立清晰的模块接口

### Phase 4: 性能优化 (1-2天)

#### 4.1 添加缓存机制
```python
# utils/cache.py
@functools.lru_cache(maxsize=128)
def calculate_network_params(network_config):
    # 缓存参数计算结果
```

#### 4.2 优化计算逻辑
- 减少不必要的数组创建
- 优化矩阵运算
- 添加进度指示器

### Phase 5: 测试与文档 (1天)

#### 5.1 添加单元测试
```python
# tests/test_chart_utils.py
def test_create_line_chart():
    # 测试图表创建功能
```

#### 5.2 更新文档
- API文档
- 使用示例
- 贡献指南

## 📈 预期收益

1. **代码量减少30%** - 通过消除重复代码
2. **性能提升50%** - 通过缓存和优化
3. **可维护性提升** - 模块化设计
4. **测试覆盖率达到80%** - 便于质量保证

## 🚀 实施步骤

1. **创建分支** - `refactor/charts`
2. **Phase 1实施** - 图表组件重构
3. **测试验证** - 确保功能正常
4. **合并主分支** - 逐步合并各阶段
5. **性能测试** - 验证优化效果

## ⚠️ 风险控制

1. **功能回归** - 每个阶段都进行充分测试
2. **性能下降** - 持续监控关键指标
3. **兼容性问题** - 保持API向后兼容

## 📅 时间安排

- **Phase 1**: 1-2天 (图表重构)
- **Phase 2**: 1天 (异常处理)
- **Phase 3**: 2-3天 (模块解耦)
- **Phase 4**: 1-2天 (性能优化)
- **Phase 5**: 1天 (测试文档)

**总计**: 6-9天完成全部重构