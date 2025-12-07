"""
层分析器单元测试
"""

import pytest
from tabs.params_calculator.layer_analyzer import LayerAnalyzer


class TestLayerAnalyzer:
    """层分析器测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.analyzer = LayerAnalyzer()
    
    def test_conv2d_analysis_basic(self):
        """测试Conv2d基本分析"""
        result = self.analyzer.conv2d_analysis(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            input_shape=(3, 224, 224),
            use_bias=True
        )
        
        # 验证返回结构
        assert result["layer_type"] == "Conv2d"
        assert "parameters" in result
        assert "flops" in result
        assert "memory_mb" in result
        
        # 验证参数量计算
        expected_params = 64 * 3 * 3 * 3 + 64  # weight + bias
        assert result["parameters"]["total"] == expected_params
        
        # 验证输出形状
        assert result["output_shape"] == (64, 224, 224)
    
    def test_conv2d_analysis_stride_padding(self):
        """测试Conv2d不同步长和填充"""
        result = self.analyzer.conv2d_analysis(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            input_shape=(3, 224, 224),
            use_bias=False
        )
        
        # 验证输出尺寸计算: (224 + 2*3 - 7) // 2 + 1 = 112
        assert result["output_shape"] == (64, 112, 112)
        
        # 验证无偏置参数
        assert result["parameters"]["bias"] == 0
        assert result["parameters"]["total"] == result["parameters"]["weight"]
    
    def test_linear_analysis_basic(self):
        """测试Linear基本分析"""
        result = self.analyzer.linear_analysis(
            in_features=512,
            out_features=1000,
            use_bias=True
        )
        
        # 验证返回结构
        assert result["layer_type"] == "Linear"
        assert result["input_features"] == 512
        assert result["output_features"] == 1000
        
        # 验证参数量计算
        expected_params = 512 * 1000 + 1000  # weight + bias
        assert result["parameters"]["total"] == expected_params
    
    def test_linear_analysis_no_bias(self):
        """测试Linear无偏置情况"""
        result = self.analyzer.linear_analysis(
            in_features=256,
            out_features=128,
            use_bias=False
        )
        
        assert result["parameters"]["bias"] == 0
        assert result["parameters"]["total"] == 256 * 128
    
    def test_attention_analysis_basic(self):
        """测试多头注意力基本分析"""
        result = self.analyzer.attention_analysis(
            d_model=512,
            num_heads=8,
            seq_len=128,
            has_qkv_bias=True
        )
        
        # 验证返回结构
        assert result["layer_type"] == "MultiHeadAttention"
        assert result["d_model"] == 512
        assert result["num_heads"] == 8
        assert result["seq_len"] == 128
        
        # 验证参数量计算
        # QKV: 3 * 512 * 512, QKV bias: 3 * 512
        # Output: 512 * 512, Output bias: 512
        expected_params = 3 * 512 * 512 + 3 * 512 + 512 * 512 + 512
        assert result["parameters"]["total"] == expected_params
    
    def test_attention_analysis_no_bias(self):
        """测试多头注意力无偏置情况"""
        result = self.analyzer.attention_analysis(
            d_model=256,
            num_heads=4,
            seq_len=64,
            has_qkv_bias=False
        )
        
        # 验证无偏置参数
        assert result["parameters"]["qkv_bias"] == 0
        assert result["parameters"]["out_bias"] == 0
    
    def test_depthwise_conv2d_analysis(self):
        """测试深度可分离卷积分析"""
        result = self.analyzer.depthwise_conv2d_analysis(
            in_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            input_shape=(32, 56, 56),
            use_bias=True
        )
        
        # 验证返回结构
        assert result["layer_type"] == "DepthwiseConv2d"
        assert result["output_shape"] == (32, 56, 56)
        
        # 验证参数量计算: in_channels * kernel_size * kernel_size + bias
        expected_params = 32 * 3 * 3 + 32
        assert result["parameters"]["total"] == expected_params
    
    def test_lstm_analysis_basic(self):
        """测试LSTM基本分析"""
        result = self.analyzer.lstm_analysis(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            bias=True,
            bidirectional=False
        )
        
        # 验证返回结构
        assert result["layer_type"] == "LSTM"
        assert result["input_size"] == 128
        assert result["hidden_size"] == 256
        assert result["num_layers"] == 2
        assert result["bidirectional"] is False
        
        # 验证有参数量
        assert result["parameters"]["total"] > 0
        assert "per_layer" in result["parameters"]
    
    def test_lstm_analysis_bidirectional(self):
        """测试LSTM双向情况"""
        result = self.analyzer.lstm_analysis(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            bias=True,
            bidirectional=True
        )
        
        assert result["bidirectional"] is True
        # 双向LSTM参数量应该是单向的2倍
        unidirectional_result = self.analyzer.lstm_analysis(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            bias=True,
            bidirectional=False
        )
        assert result["parameters"]["total"] == 2 * unidirectional_result["parameters"]["total"]
    
    def test_layernorm_analysis(self):
        """测试LayerNorm分析"""
        result = self.analyzer.layernorm_analysis(
            normalized_shape=512,
            input_shape=(2, 10, 512)
        )
        
        # 验证返回结构
        assert result["layer_type"] == "LayerNorm"
        assert result["normalized_shape"] == 512
        
        # 验证参数量: gamma + beta
        assert result["parameters"]["total"] == 2 * 512
        assert result["parameters"]["gamma"] == 512
        assert result["parameters"]["beta"] == 512
    
    def test_embedding_analysis(self):
        """测试Embedding分析"""
        result = self.analyzer.embedding_analysis(
            num_embeddings=10000,
            embedding_dim=300
        )
        
        # 验证返回结构
        assert result["layer_type"] == "Embedding"
        assert result["num_embeddings"] == 10000
        assert result["embedding_dim"] == 300
        
        # 验证参数量计算
        expected_params = 10000 * 300
        assert result["parameters"]["total"] == expected_params
        
        # 验证FLOPs为0（查表操作）
        assert result["flops"]["total"] == 0
    
    def test_batchnorm2d_analysis(self):
        """测试BatchNorm2d分析"""
        result = self.analyzer.batchnorm2d_analysis(
            num_features=64,
            input_shape=(64, 56, 56)
        )
        
        # 验证返回结构
        assert result["layer_type"] == "BatchNorm2d"
        assert result["num_features"] == 64
        
        # 验证参数量: gamma + beta
        assert result["parameters"]["total"] == 2 * 64
        assert result["parameters"]["gamma"] == 64
        assert result["parameters"]["beta"] == 64
        
        # 验证FLOPs计算: 4 * elements
        expected_flops = 4 * 64 * 56 * 56
        assert result["flops"]["total"] == expected_flops
    
    def test_flops_formatting(self):
        """测试FLOPs格式化"""
        # 测试大数值格式化
        result = self.analyzer.conv2d_analysis(
            in_channels=3,
            out_channels=512,
            kernel_size=7,
            stride=2,
            padding=3,
            input_shape=(3, 224, 224),
            use_bias=True
        )
        
        # 验证FLOPs格式化
        flops_readable = result["flops"]["flops_readable"]
        assert "G" in flops_readable or "M" in flops_readable
        
        # 验证MACs格式化
        macs_readable = result["flops"]["macs_readable"]
        assert "M" in macs_readable or "K" in macs_readable
    
    def test_memory_calculation(self):
        """测试内存计算"""
        result = self.analyzer.conv2d_analysis(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            input_shape=(3, 224, 224),
            use_bias=True
        )
        
        # 验证内存计算
        assert result["memory_mb"]["parameters"] > 0
        assert result["memory_mb"]["forward"] > 0
        assert result["memory_mb"]["backward"] > 0
        assert result["memory_mb"]["total"] > 0
        
        # 验证反向传播内存应该大于前向传播
        assert result["memory_mb"]["backward"] > result["memory_mb"]["forward"]