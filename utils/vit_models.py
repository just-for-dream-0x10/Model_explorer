"""
Vision Transformer (ViT) 模型定义模块
Vision Transformer Models

实现ViT的核心组件，用于教学演示
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Patch Embedding层

    将图像切分成patches，然后通过线性投影变成embeddings

    例如：
    - 输入图像：[B, 3, 224, 224]
    - Patch大小：16×16
    - 输出：[B, 196, 768]（196 = 14×14个patches）
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 方法1：使用卷积实现（等价于线性投影）
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 投影：[B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.projection(x)  # [B, 768, 14, 14]

        # 展平：[B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)  # [B, 768, 196]

        # 转置：[B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [B, 196, 768]

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制

    核心计算：
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        # Q、K、V的线性投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: [B, num_patches+1, embed_dim]
        B, N, C = x.shape

        # 生成Q、K、V
        qkv = self.qkv(x)  # [B, N, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn_weights = attn  # 保存用于可视化
        attn = self.dropout(attn)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        x = self.proj(x)
        x = self.dropout(x)

        if return_attention:
            return x, attn_weights
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder块

    结构：
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. MLP + Residual + LayerNorm
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        # Self-Attention + Residual
        if return_attention:
            attn_output, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_output
        else:
            x = x + self.attn(self.norm1(x))
            attn_weights = None

        # MLP + Residual
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)

    结构：
    1. Patch Embedding
    2. Position Embedding (可学习)
    3. [CLS] Token
    4. N × Transformer Encoder Blocks
    5. Classification Head
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # [CLS] Token（可学习）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.shape[0]

        # Patch Embedding: [B, 3, 224, 224] -> [B, 196, 768]
        x = self.patch_embed(x)

        # 添加[CLS] Token: [B, 196, 768] -> [B, 197, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加Position Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder Blocks
        attn_weights_list = []
        for block in self.blocks:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attn_weights_list.append(attn_weights)
            else:
                x = block(x)

        # LayerNorm
        x = self.norm(x)

        # 取[CLS] Token的输出
        cls_output = x[:, 0]

        # Classification
        logits = self.head(cls_output)

        if return_attention:
            return logits, attn_weights_list
        return logits


def create_vit_tiny(img_size=224, num_classes=1000):
    """创建ViT-Tiny模型（用于快速演示）"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
    )


def create_vit_small(img_size=224, num_classes=1000):
    """创建ViT-Small模型"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )


def create_vit_base(img_size=224, num_classes=1000):
    """创建ViT-Base模型（标准配置）"""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
    )


def get_vit_info(model_name="vit_tiny"):
    """
    获取ViT模型信息

    Args:
        model_name: "vit_tiny", "vit_small", "vit_base"

    Returns:
        info: 模型信息字典
    """
    configs = {
        "vit_tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "patch_size": 16,
            "params_estimate": "5.7M",
        },
        "vit_small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "patch_size": 16,
            "params_estimate": "22M",
        },
        "vit_base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 16,
            "params_estimate": "86M",
        },
    }

    if model_name not in configs:
        raise ValueError(f"未知模型: {model_name}")

    config = configs[model_name]

    return {
        "name": model_name,
        "config": config,
        "architecture": {
            "patch_embedding": f"Conv2d(3, {config['embed_dim']}, kernel_size={config['patch_size']}, stride={config['patch_size']})",
            "position_embedding": f"Learnable [1, 197, {config['embed_dim']}]",
            "cls_token": f"Learnable [1, 1, {config['embed_dim']}]",
            "transformer_blocks": f"{config['depth']} blocks",
            "attention": f"{config['num_heads']} heads, head_dim={config['embed_dim']//config['num_heads']}",
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Vision Transformer (ViT) 模型测试")
    print("=" * 60)

    # 测试Patch Embedding
    print("\n### Patch Embedding测试 ###")
    patch_embed = PatchEmbedding(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768
    )
    x = torch.randn(2, 3, 224, 224)
    patches = patch_embed(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {patches.shape}")
    print(f"Patches数量: {patch_embed.num_patches}")

    # 测试Multi-Head Self-Attention
    print("\n### Multi-Head Self-Attention测试 ###")
    attn = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)  # [B, N, C] (包含CLS token)
    output, attn_weights = attn(x, return_attention=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    # 测试ViT模型
    print("\n### ViT模型测试 ###")
    for model_name in ["vit_tiny", "vit_small", "vit_base"]:
        print(f"\n--- {model_name.upper()} ---")

        if model_name == "vit_tiny":
            model = create_vit_tiny()
        elif model_name == "vit_small":
            model = create_vit_small()
        else:
            model = create_vit_base()

        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,} ({params/1e6:.1f}M)")

        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        try:
            output = model(x)
            print(f"✅ 前向传播成功，输出形状: {output.shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
