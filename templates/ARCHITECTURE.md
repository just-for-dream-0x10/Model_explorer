# 神经网络模板系统 - 架构设计文档

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI                           │
│          (architecture_designer.py)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 调用
                       ▼
┌─────────────────────────────────────────────────────────┐
│              TemplateLoader                             │
│         (templates/template_loader.py)                  │
│                                                          │
│  - get_all_templates()                                  │
│  - get_template(id)                                     │
│  - get_templates_by_category()                          │
│  - search_templates()                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ 加载
                       ▼
┌─────────────────────────────────────────────────────────┐
│           JSON 模板配置文件                              │
│        (templates/configs/*.json)                       │
│                                                          │
│  - mnist_cnn.json                                       │
│  - cifar_cnn.json                                       │
│  - simple_mlp.json                                      │
│  - lenet.json                                           │
│  - alexnet_like.json                                    │
│  - vgg_like.json                                        │
│  - ... (更多)                                           │
└─────────────────────────────────────────────────────────┘
```

## 📦 核心组件

### 1. LayerConfig (数据类)

统一的层配置数据结构，用于在模板系统和架构设计器之间传递层信息。

```python
@dataclass
class LayerConfig:
    # 基本信息
    layer_type: str              # 层类型 (Conv2d, Linear, ReLU等)
    name: str                    # 层名称
    params: Dict[str, Any]       # 层参数
    
    # 运行时计算
    output_shape: Optional[Tuple[int, ...]]  # 输出形状
    memory: float = 0.0          # 内存占用 (MB)
    param_count: int = 0         # 参数数量
    flops: int = 0               # 浮点运算次数
    
    # 问题检测
    has_issues: bool = False     # 是否有错误
    warnings: List[str]          # 警告列表
    issues: List[str]            # 错误列表
    recommendations: List[str]   # 建议列表
```

**关键方法**:
- `to_dict()`: 导出为字典 (用于保存配置)
- `from_dict()`: 从字典创建 (用于加载配置)

### 2. NetworkTemplate (数据类)

网络模板的完整描述，包含元数据和层配置。

```python
@dataclass
class NetworkTemplate:
    id: str                      # 唯一标识符
    name: str                    # 显示名称
    description: str             # 描述
    category: str                # 分类 (CNN, MLP, Autoencoder等)
    input_shape: Tuple[int, ...]  # 输入形状
    layers: List[Dict[str, Any]]  # 层配置列表
    
    # 可选元数据
    tags: List[str]              # 标签
    difficulty: str              # 难度等级
    use_cases: List[str]         # 使用场景
    icon: str                    # 显示图标
```

**关键方法**:
- `to_layer_configs()`: 将层配置转换为 LayerConfig 对象列表

### 3. TemplateLoader (管理类)

模板加载和管理的核心类。

```python
class TemplateLoader:
    def __init__(self, templates_dir: Optional[str] = None)
    def get_template(self, template_id: str) -> Optional[NetworkTemplate]
    def get_all_templates(self) -> List[NetworkTemplate]
    def get_templates_by_category(self, category: str) -> List[NetworkTemplate]
    def get_templates_by_difficulty(self, difficulty: str) -> List[NetworkTemplate]
    def get_categories(self) -> List[str]
    def search_templates(self, keyword: str) -> List[NetworkTemplate]
```

## 🔄 数据流

### 加载模板流程

```
1. 用户点击模板按钮
         ↓
2. Streamlit 调用 loader.get_template(template_id)
         ↓
3. TemplateLoader 读取 JSON 文件
         ↓
4. 解析 JSON 创建 NetworkTemplate 对象
         ↓
5. 调用 template.to_layer_configs()
         ↓
6. 返回 List[LayerConfig]
         ↓
7. 设置 st.session_state.layers = layer_configs
         ↓
8. 设置 st.session_state.input_shape = template.input_shape
         ↓
9. 重新渲染 UI (st.rerun())
         ↓
10. 显示加载的网络架构
```

### 导出配置流程

```
1. 用户点击"下载配置"
         ↓
2. 调用 export_network_config(layers, input_shape)
         ↓
3. 遍历每个 LayerConfig，调用 layer.to_dict()
         ↓
4. 生成 JSON 字符串
         ↓
5. 通过 st.download_button() 提供下载
```

### 导入配置流程

```
1. 用户上传 JSON 文件
         ↓
2. 调用 import_network_config(json_str)
         ↓
3. JSON.parse() 解析字符串
         ↓
4. 遍历每个层字典，调用 LayerConfig.from_dict()
         ↓
5. 返回 (layers, input_shape)
         ↓
6. 更新 session_state
         ↓
7. 重新渲染 UI
```

## 🎨 UI 集成

### 模板库界面结构

```
🚀 神经网络模板库 (Expander)
├── 筛选选项
│   ├── 📂 按分类筛选 (Selectbox)
│   ├── 📊 按难度筛选 (Selectbox)
│   └── 🔍 搜索模板 (Text Input)
│
└── 模板展示
    ├── 📁 CNN
    │   ├── [按钮] 📱 MNIST CNN 🟢
    │   ├── [按钮] 🖼️ CIFAR-10 CNN 🟡
    │   └── [按钮] 🏆 AlexNet-like 🟡
    │
    ├── 📁 MLP
    │   ├── [按钮] 🧠 简单MLP 🟢
    │   └── [按钮] 🧬 深度MLP 🟡
    │
    └── 📁 Autoencoder
        ├── [按钮] 🔄 自编码器 🟡
        └── [按钮] 🔁 卷积自编码器 🔴
```

### 难度颜色编码

- 🟢 绿色 = `beginner` (入门)
- 🟡 黄色 = `intermediate` (中级)
- 🔴 红色 = `advanced` (高级)

## 🔧 扩展点

### 添加新分类

只需在 JSON 中使用新的 `category` 值，系统会自动识别：

```json
{
  "category": "Transformer",
  "name": "BERT-like",
  ...
}
```

`TemplateLoader.get_categories()` 会自动返回包含新分类的列表。

### 添加新层类型

1. 在 `architecture_designer.py` 的 `create_layer_from_config()` 中添加新的层类型处理
2. 在模板 JSON 中使用新的层类型
3. 系统会自动支持

示例：添加 `AvgPool2d` 支持

```python
# 在 create_layer_from_config() 中添加
elif layer_type == "AvgPool2d":
    layer = nn.AvgPool2d(params['kernel_size'], stride=params.get('stride', params['kernel_size']))
    # 计算输出形状
    B, C, H, W = input_shape
    H_out = (H - params['kernel_size']) // params.get('stride', params['kernel_size']) + 1
    W_out = (W - params['kernel_size']) // params.get('stride', params['kernel_size']) + 1
    config.output_shape = (B, C, H_out, W_out)
```

然后在模板中使用：

```json
{
  "layer_type": "AvgPool2d",
  "name": "avgpool1",
  "params": {
    "kernel_size": 2,
    "stride": 2
  }
}
```

### 自定义模板加载器

可以继承 `TemplateLoader` 创建自定义加载器：

```python
class CustomTemplateLoader(TemplateLoader):
    def __init__(self):
        super().__init__()
        # 添加自定义模板目录
        self._load_custom_templates()
    
    def _load_custom_templates(self):
        # 从数据库、API或其他源加载模板
        pass
```

## 📊 性能考虑

### 模板加载性能

- **冷启动**: 首次加载时读取所有 JSON 文件 (~12 个文件)
- **缓存**: 加载后缓存在内存中 (`TemplateLoader.templates` 字典)
- **查询**: O(1) 通过 ID 查找，O(n) 筛选和搜索

### 优化建议

1. **延迟加载**: 在用户打开模板库时才初始化 `TemplateLoader`
2. **索引**: 为分类和难度建立索引以加速筛选
3. **压缩**: 对大型模板使用压缩存储

## 🧪 测试

### 单元测试模板

```python
def test_template_loader():
    loader = TemplateLoader()
    
    # 测试加载所有模板
    templates = loader.get_all_templates()
    assert len(templates) > 0
    
    # 测试按ID获取
    template = loader.get_template('mnist_cnn')
    assert template is not None
    assert template.name == "MNIST CNN"
    
    # 测试筛选
    cnn_templates = loader.get_templates_by_category('CNN')
    assert len(cnn_templates) > 0
    
    # 测试搜索
    results = loader.search_templates('mnist')
    assert len(results) > 0
```

### 验证 JSON 模板

```python
def validate_template(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    # 检查必填字段
    required_fields = ['id', 'name', 'description', 'category', 'input_shape', 'layers']
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    # 检查层配置
    for layer in data['layers']:
        assert 'layer_type' in layer
        assert 'name' in layer
        assert 'params' in layer
```

## 📝 最佳实践总结

1. **模块化**: 每个模板一个独立的 JSON 文件
2. **命名规范**: 使用描述性的 ID 和名称
3. **完整性**: 包含所有必要的元数据（标签、使用场景等）
4. **验证**: 添加新模板后进行测试
5. **文档**: 在 README.md 中记录新模板
6. **版本控制**: 使用 Git 跟踪模板变更

## 🚀 未来扩展

### 计划功能

1. **模板继承**: 支持基于现有模板创建变体
2. **在线模板库**: 从云端下载社区贡献的模板
3. **模板评分**: 用户可以对模板进行评分和评论
4. **自动生成**: 基于数据集特征自动推荐模板
5. **参数搜索**: 自动调优模板中的超参数

### 扩展示例：模板继承

```json
{
  "id": "mnist_cnn_v2",
  "extends": "mnist_cnn",
  "modifications": {
    "replace_layer": {
      "name": "fc1",
      "with": {
        "layer_type": "Linear",
        "name": "fc1",
        "params": {"in_features": 3136, "out_features": 256}
      }
    },
    "insert_after": {
      "after": "conv2",
      "layers": [
        {
          "layer_type": "Dropout",
          "name": "dropout_extra",
          "params": {"p": 0.3}
        }
      ]
    }
  }
}
```

## 🤝 贡献指南

添加新模板时，请确保：

1. ✅ JSON 格式正确（使用 JSON linter 验证）
2. ✅ 所有必填字段都已填写
3. ✅ 层配置完整且正确
4. ✅ 输入形状与第一层匹配
5. ✅ 通过测试验证
6. ✅ 更新 README.md 文档
7. ✅ 提交 Pull Request

---

**维护者**: Neural Network Math Explorer Team  
**最后更新**: 2024
