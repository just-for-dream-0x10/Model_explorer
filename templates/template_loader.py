"""
模板加载器 - 负责加载和管理所有神经网络模板
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class LayerConfig:
    """层配置"""

    layer_type: str
    name: str
    params: Dict[str, Any]
    output_shape: Optional[Tuple[int, ...]] = None
    memory: float = 0.0
    has_issues: bool = False
    warnings: List[str] = field(default_factory=list)
    param_count: int = 0
    flops: int = 0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于导出）"""
        return {"layer_type": self.layer_type, "name": self.name, "params": self.params}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LayerConfig":
        """从字典创建（用于导入）"""
        return LayerConfig(
            layer_type=data["layer_type"], name=data["name"], params=data["params"]
        )


@dataclass
class NetworkTemplate:
    """神经网络模板"""

    id: str  # 模板ID，如 "mnist_cnn"
    name: str  # 显示名称，如 "MNIST CNN"
    description: str  # 描述
    category: str  # 分类：CNN, RNN, Transformer, GAN等
    input_shape: Tuple[int, ...]  # 输入形状
    layers: List[Dict[str, Any]]  # 层配置列表
    tags: List[str] = field(default_factory=list)  # 标签
    difficulty: str = "beginner"  # 难度：beginner, intermediate, advanced
    use_cases: List[str] = field(default_factory=list)  # 使用场景
    icon: str = "🧠"  # 显示图标

    def to_layer_configs(self) -> List[LayerConfig]:
        """转换为 LayerConfig 对象列表"""
        return [
            LayerConfig(
                layer_type=layer["layer_type"],
                name=layer["name"],
                params=layer["params"],
            )
            for layer in self.layers
        ]


class TemplateLoader:
    """模板加载器"""

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "configs"
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, NetworkTemplate] = {}
        self._load_all_templates()

    def _load_all_templates(self):
        """加载所有模板"""
        if not self.templates_dir.exists():
            return

        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    template = NetworkTemplate(**data)
                    self.templates[template.id] = template
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")

    def get_template(self, template_id: str) -> Optional[NetworkTemplate]:
        """获取指定模板"""
        return self.templates.get(template_id)

    def get_all_templates(self) -> List[NetworkTemplate]:
        """获取所有模板"""
        return list(self.templates.values())

    def get_templates_by_category(self, category: str) -> List[NetworkTemplate]:
        """按分类获取模板"""
        return [t for t in self.templates.values() if t.category == category]

    def get_templates_by_difficulty(self, difficulty: str) -> List[NetworkTemplate]:
        """按难度获取模板"""
        return [t for t in self.templates.values() if t.difficulty == difficulty]

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return sorted(set(t.category for t in self.templates.values()))

    def search_templates(self, keyword: str) -> List[NetworkTemplate]:
        """搜索模板"""
        keyword = keyword.lower()
        return [
            t
            for t in self.templates.values()
            if keyword in t.name.lower()
            or keyword in t.description.lower()
            or any(keyword in tag.lower() for tag in t.tags)
        ]
