"""
架构模板自适应计算器
Architecture Template Adaptive Calculator

为架构设计器提供动态计算模板参数的功能
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class TemplateCalculator:
    """模板自适应计算器

    根据用户输入的形状自动计算模板中的参数
    """

    @staticmethod
    def calculate_flattened_size(shape: Tuple[int, ...]) -> int:
        """
        计算flatten后的特征数

        Args:
            shape: 输入形状

        Returns:
            flatten后的特征数
        """
        return int(np.prod(shape))

    @staticmethod
    def suggest_fc_features(input_features: int, output_classes: int = 10) -> List[int]:
        """
        建议全连接层的特征数

        Args:
            input_features: 输入特征数
            output_classes: 输出类别数

        Returns:
            建议的各层特征数列表
        """
        # 基于输入特征数动态建议
        if input_features < 1000:
            # 小网络
            return [max(128, input_features // 4), output_classes]
        elif input_features < 10000:
            # 中等网络
            return [
                max(256, input_features // 8),
                max(128, input_features // 16),
                output_classes,
            ]
        else:
            # 大网络
            return [
                max(512, input_features // 16),
                max(256, input_features // 32),
                max(128, input_features // 64),
                output_classes,
            ]

    @staticmethod
    def suggest_conv_channels(input_shape: Tuple[int, int, int]) -> List[int]:
        """
        建议卷积层的通道数

        Args:
            input_shape: (C, H, W) 输入形状

        Returns:
            建议的各层通道数列表
        """
        _, H, W = input_shape
        input_channels = input_shape[0]

        # 基于输入尺寸动态建议
        if H <= 32:
            # 小图像
            base_channels = max(32, input_channels * 2)
            return [base_channels, base_channels * 2, base_channels * 4]
        elif H <= 64:
            # 中等图像
            base_channels = max(64, input_channels * 2)
            return [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
            ]
        else:
            # 大图像
            base_channels = max(64, input_channels * 2)
            return [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ]

    @staticmethod
    def calculate_output_shape(
        input_shape: Tuple[int, ...], layer_type: str, layer_params: Dict
    ) -> Tuple[int, ...]:
        """
        计算层的输出形状

        Args:
            input_shape: 输入形状
            layer_type: 层类型
            layer_params: 层参数

        Returns:
            输出形状
        """
        if layer_type == "Conv2d":
            C_in, H_in, W_in = input_shape
            kernel_size = layer_params["kernel_size"]
            stride = layer_params.get("stride", 1)
            padding = layer_params.get("padding", 0)
            out_channels = layer_params["out_channels"]

            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            return (out_channels, H_out, W_out)

        elif layer_type == "MaxPool2d":
            C_in, H_in, W_in = input_shape
            kernel_size = layer_params["kernel_size"]
            stride = layer_params.get("stride", kernel_size)
            padding = layer_params.get("padding", 0)

            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            return (C_in, H_out, W_out)

        elif layer_type == "Linear":
            B = input_shape[0]
            out_features = layer_params["out_features"]
            return (B, out_features)

        elif layer_type in ["ReLU", "BatchNorm2d", "Dropout"]:
            return input_shape

        elif layer_type == "Flatten":
            B = input_shape[0]
            flat_size = TemplateCalculator.calculate_flattened_size(input_shape[1:])
            return (B, flat_size)

        else:
            return input_shape

    @staticmethod
    def create_mnist_template(input_shape: Tuple[int, ...]) -> List[Dict]:
        """
        创建自适应的MNIST模板

        Args:
            input_shape: 输入形状

        Returns:
            层配置列表
        """
        from tabs.architecture_designer import LayerConfig

        # 计算flatten后的特征数
        after_conv_shape = (
            64,
            input_shape[2] // 4,
            input_shape[3] // 4,
        )  # 假设经过2次2x2池化
        flattened_size = TemplateCalculator.calculate_flattened_size(after_conv_shape)

        # 建议全连接层特征数
        fc_features = TemplateCalculator.suggest_fc_features(flattened_size, 10)

        template = [
            LayerConfig(
                "Conv2d",
                "conv1",
                {
                    "in_channels": input_shape[1],
                    "out_channels": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ),
            LayerConfig("ReLU", "relu1", {}),
            LayerConfig("MaxPool2d", "pool1", {"kernel_size": 2, "stride": 2}),
            LayerConfig(
                "Conv2d",
                "conv2",
                {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ),
            LayerConfig("ReLU", "relu2", {}),
            LayerConfig("MaxPool2d", "pool2", {"kernel_size": 2, "stride": 2}),
            LayerConfig("Flatten", "flatten", {}),
            LayerConfig(
                "Linear",
                "fc1",
                {"in_features": flattened_size, "out_features": fc_features[0]},
            ),
            LayerConfig("ReLU", "relu3", {}),
            LayerConfig(
                "Linear", "fc2", {"in_features": fc_features[0], "out_features": 10}
            ),
        ]

        return template

    @staticmethod
    def create_cifar_template(input_shape: Tuple[int, ...]) -> List[Dict]:
        """
        创建自适应的CIFAR模板

        Args:
            input_shape: 输入形状

        Returns:
            层配置列表
        """
        from tabs.architecture_designer import LayerConfig

        # 计算flatten后的特征数
        after_conv_shape = (
            128,
            input_shape[2] // 4,
            input_shape[3] // 4,
        )  # 假设经过2次2x2池化
        flattened_size = TemplateCalculator.calculate_flattened_size(after_conv_shape)

        # 建议全连接层特征数
        fc_features = TemplateCalculator.suggest_fc_features(flattened_size, 10)

        template = [
            LayerConfig(
                "Conv2d",
                "conv1",
                {
                    "in_channels": input_shape[1],
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ),
            LayerConfig("BatchNorm2d", "bn1", {"num_features": 64}),
            LayerConfig("ReLU", "relu1", {}),
            LayerConfig(
                "Conv2d",
                "conv2",
                {
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ),
            LayerConfig("BatchNorm2d", "bn2", {"num_features": 128}),
            LayerConfig("ReLU", "relu2", {}),
            LayerConfig("MaxPool2d", "pool1", {"kernel_size": 2, "stride": 2}),
            LayerConfig("Flatten", "flatten", {}),
            LayerConfig(
                "Linear",
                "fc1",
                {"in_features": flattened_size, "out_features": fc_features[0]},
            ),
            LayerConfig("ReLU", "relu3", {}),
            LayerConfig("Dropout", "dropout", {"p": 0.5}),
            LayerConfig(
                "Linear", "fc2", {"in_features": fc_features[0], "out_features": 10}
            ),
        ]

        return template

    @staticmethod
    def create_mlp_template(input_shape: Tuple[int, ...]) -> List[Dict]:
        """
        创建自适应的MLP模板

        Args:
            input_shape: 输入形状

        Returns:
            层配置列表
        """
        from tabs.architecture_designer import LayerConfig

        input_features = input_shape[1] if len(input_shape) == 2 else input_shape[0]

        # 建议各层特征数
        layer_features = TemplateCalculator.suggest_fc_features(input_features, 10)

        template = []

        # 添加输入层到第一个隐藏层
        template.append(
            LayerConfig(
                "Linear",
                "fc1",
                {"in_features": input_features, "out_features": layer_features[0]},
            )
        )
        template.append(LayerConfig("ReLU", "relu1", {}))
        template.append(LayerConfig("Dropout", "dropout1", {"p": 0.2}))

        # 添加中间层
        for i in range(1, len(layer_features) - 1):
            template.append(
                LayerConfig(
                    "Linear",
                    f"fc{i+1}",
                    {
                        "in_features": layer_features[i - 1],
                        "out_features": layer_features[i],
                    },
                )
            )
            template.append(LayerConfig("ReLU", f"relu{i+1}", {}))
            template.append(LayerConfig("Dropout", f"dropout{i+1}", {"p": 0.2}))

        # 添加输出层
        template.append(
            LayerConfig(
                "Linear",
                f"fc{len(layer_features)}",
                {"in_features": layer_features[-2], "out_features": layer_features[-1]},
            )
        )

        return template
