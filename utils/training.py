"""
训练模拟工具
"""

import numpy as np


def simulate_training(**kwargs):
    """
    模拟神经网络训练过程
    
    Args:
        **kwargs: 训练参数（learning_rate, epochs, etc.）
        
    Returns:
        训练历史数据
    """
    learning_rate = kwargs.get("learning_rate", 0.001)
    epochs = kwargs.get("epochs", 100)
    batch_size = kwargs.get("batch_size", 32)
    
    # 模拟损失曲线
    train_loss = []
    val_loss = []
    
    initial_loss = 2.0
    
    for epoch in range(epochs):
        # 训练损失：指数衰减 + 随机噪声
        decay_rate = 0.95
        noise = np.random.normal(0, 0.02)
        loss = initial_loss * (decay_rate ** epoch) + noise
        train_loss.append(max(loss, 0.01))  # 确保损失不为负
        
        # 验证损失：稍高于训练损失
        val_noise = np.random.normal(0, 0.03)
        val_loss.append(max(loss * 1.1 + val_noise, 0.01))
    
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epochs": list(range(epochs)),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
