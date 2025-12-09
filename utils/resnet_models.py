"""
ResNet残差连接网络定义模块
ResNet Models with Residual Connections

用于对比有无残差连接的梯度流差异
"""

import torch
import torch.nn as nn


class PlainBlock(nn.Module):
    """
    普通卷积块（无残差连接）

    结构：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class ResidualBlock(nn.Module):
    """
    残差块（有残差连接）

    结构：Conv -> BN -> ReLU -> Conv -> BN -> (+shortcut) -> ReLU
    核心：y = F(x) + x （梯度高速公路）
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut连接（处理维度不匹配）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x  # 保存输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 关键：残差连接
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class PlainNet(nn.Module):
    """
    普通深层网络（无残差连接）

    用于对比：深度增加时梯度会消失
    """

    def __init__(self, num_blocks=10, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 堆叠普通块
        self.layers = self._make_layer(PlainBlock, 64, num_blocks)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet(nn.Module):
    """
    ResNet网络（有残差连接）

    验证：残差连接如何保持梯度流
    """

    def __init__(self, num_blocks=10, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 堆叠残差块
        self.layers = self._make_layer(ResidualBlock, 64, num_blocks)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_resnet_comparison(num_blocks=10, num_classes=10):
    """
    获取对比网络（普通网络 vs ResNet）

    Args:
        num_blocks: 块的数量（深度）
        num_classes: 分类数

    Returns:
        plain_net: 普通网络
        resnet: ResNet网络
        info: 网络信息字典
    """
    plain_net = PlainNet(num_blocks=num_blocks, num_classes=num_classes)
    resnet = ResNet(num_blocks=num_blocks, num_classes=num_classes)

    # 计算参数量
    plain_params = sum(p.numel() for p in plain_net.parameters())
    resnet_params = sum(p.numel() for p in resnet.parameters())

    info = {
        "num_blocks": num_blocks,
        "num_classes": num_classes,
        "plain_params": plain_params,
        "resnet_params": resnet_params,
        "input_size": (1, 3, 224, 224),
        "architecture": {
            "plain": "Conv -> BN -> ReLU -> Conv -> BN -> ReLU (无残差)",
            "resnet": "Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU (有残差)",
        },
    }

    return plain_net, resnet, info


class TinyPlainNet(nn.Module):
    """
    简化的普通网络（用于快速演示）
    纯全连接层版本，便于理解梯度流
    """

    def __init__(self, num_layers=20, hidden_dim=128):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(10, hidden_dim))
        layers.append(nn.ReLU())

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TinyResNet(nn.Module):
    """
    简化的ResNet（用于快速演示）
    纯全连接层版本，带残差连接
    """

    def __init__(self, num_layers=20, hidden_dim=128):
        super().__init__()

        # 输入层
        self.input_layer = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU())

        # 残差块
        self.res_blocks = nn.ModuleList()
        for _ in range((num_layers - 2) // 2):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)

        # 残差连接
        for block in self.res_blocks:
            identity = x
            x = block(x)
            x = x + identity  # 关键：残差连接
            x = self.relu(x)

        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("ResNet残差连接模型测试")
    print("=" * 60)

    # 测试完整版网络
    print("\n### 完整版网络（卷积）###")
    plain_net, resnet, info = get_resnet_comparison(num_blocks=10)
    print(f"块数量: {info['num_blocks']}")
    print(f"普通网络参数量: {info['plain_params']:,}")
    print(f"ResNet参数量: {info['resnet_params']:,}")

    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)

    try:
        y_plain = plain_net(x)
        print(f"✅ 普通网络前向传播成功，输出形状: {y_plain.shape}")
    except Exception as e:
        print(f"❌ 普通网络前向传播失败: {e}")

    try:
        y_resnet = resnet(x)
        print(f"✅ ResNet前向传播成功，输出形状: {y_resnet.shape}")
    except Exception as e:
        print(f"❌ ResNet前向传播失败: {e}")

    # 测试简化版网络
    print("\n### 简化版网络（全连接）###")
    tiny_plain = TinyPlainNet(num_layers=20)
    tiny_resnet = TinyResNet(num_layers=20)

    plain_params = sum(p.numel() for p in tiny_plain.parameters())
    resnet_params = sum(p.numel() for p in tiny_resnet.parameters())

    print(f"简化普通网络参数量: {plain_params:,}")
    print(f"简化ResNet参数量: {resnet_params:,}")

    # 测试前向传播
    x = torch.randn(1, 10)

    try:
        y_plain = tiny_plain(x)
        print(f"✅ 简化普通网络前向传播成功，输出形状: {y_plain.shape}")
    except Exception as e:
        print(f"❌ 简化普通网络前向传播失败: {e}")

    try:
        y_resnet = tiny_resnet(x)
        print(f"✅ 简化ResNet前向传播成功，输出形状: {y_resnet.shape}")
    except Exception as e:
        print(f"❌ 简化ResNet前向传播失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
