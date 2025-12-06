"""网络分析器核心模块

提供神经网络架构分析的核心功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..exceptions import (
    InvalidLayerConfigError,
    ComputationError,
    DataValidationError
)


class NetworkAnalyzer:
    """网络分析器
    
    提供神经网络架构的统一分析接口。
    """
    
    def __init__(self, input_shape: Tuple[int, ...]):
        """初始化网络分析器
        
        Args:
            input_shape: 输入张量形状
            
        Raises:
            DataValidationError: 当输入形状无效时
        """
        if not input_shape or len(input_shape) < 3:
            raise DataValidationError(
                "input_shape", 
                input_shape, 
                "长度≥3的元组"
            )
        
        self.input_shape = input_shape
        self.layers: List[Dict[str, Any]] = []
        self._analysis_cache: Dict[str, Any] = {}
    
    def add_layer(self, layer_config: Dict[str, Any]) -> None:
        """添加网络层配置
        
        Args:
            layer_config: 层配置字典
            
        Raises:
            InvalidLayerConfigError: 当层配置无效时
        """
        self._validate_layer_config(layer_config)
        self.layers.append(layer_config)
        self._clear_cache()
    
    def _validate_layer_config(self, layer_config: Dict[str, Any]) -> None:
        """验证层配置
        
        Args:
            layer_config: 层配置字典
            
        Raises:
            InvalidLayerConfigError: 当配置无效时
        """
        required_fields = ['layer_type', 'params']
        
        for field in required_fields:
            if field not in layer_config:
                raise InvalidLayerConfigError(
                    layer_type=layer_config.get('layer_type', 'unknown'),
                    param_name=field,
                    param_value='missing',
                    expected_type='field'
                )
    
    def _clear_cache(self) -> None:
        """清除分析缓存"""
        self._analysis_cache.clear()
    
    def analyze_network(self) -> Dict[str, Any]:
        """分析整个网络
        
        Returns:
            包含网络分析结果的字典
            
        Raises:
            ComputationError: 当分析过程中出现错误时
        """
        if not self.layers:
            raise DataValidationError(
                "layers", 
                self.layers, 
                "非空列表"
            )
        
        cache_key = str(self.layers)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        try:
            result = self._perform_analysis()
            self._analysis_cache[cache_key] = result
            return result
        except Exception as e:
            raise ComputationError(
                operation="网络分析",
                error_details=str(e)
            ) from e
    
    def _perform_analysis(self) -> Dict[str, Any]:
        """执行网络分析
        
        Returns:
            分析结果字典
        """
        total_params = 0
        total_flops = 0
        total_memory = 0.0
        current_shape = self.input_shape
        
        layer_details = []
        
        for i, layer in enumerate(self.layers):
            layer_result = self._analyze_layer(layer, current_shape)
            
            total_params += layer_result['parameters']
            total_flops += layer_result['flops']
            total_memory += layer_result['memory_mb']
            current_shape = layer_result['output_shape']
            
            layer_details.append({
                'layer_index': i,
                'layer_type': layer['layer_type'],
                **layer_result
            })
        
        return {
            'total_parameters': total_params,
            'total_flops': total_flops,
            'total_memory_mb': total_memory,
            'final_shape': current_shape,
            'layer_details': layer_details,
            'input_shape': self.input_shape,
            'num_layers': len(self.layers)
        }
    
    def _analyze_layer(
        self, 
        layer: Dict[str, Any], 
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """分析单个层
        
        Args:
            layer: 层配置
            input_shape: 输入形状
            
        Returns:
            层分析结果
        """
        layer_type = layer['layer_type']
        params = layer['params']
        
        if layer_type == 'conv2d':
            return self._analyze_conv2d(params, input_shape)
        elif layer_type == 'linear':
            return self._analyze_linear(params, input_shape)
        elif layer_type == 'maxpool2d':
            return self._analyze_maxpool2d(params, input_shape)
        else:
            # 默认处理
            return {
                'parameters': 0,
                'flops': 0,
                'memory_mb': 0.0,
                'output_shape': input_shape
            }
    
    def _analyze_conv2d(
        self, 
        params: Dict[str, Any], 
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """分析卷积层
        
        Args:
            params: 卷积层参数
            input_shape: 输入形状
            
        Returns:
            卷积层分析结果
        """
        # 验证参数
        required_params = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
        for param in required_params:
            if param not in params:
                raise InvalidLayerConfigError(
                    layer_type='conv2d',
                    param_name=param,
                    param_value='missing',
                    expected_type='int'
                )
        
        in_channels = params['in_channels']
        out_channels = params['out_channels']
        kernel_size = params['kernel_size']
        stride = params['stride']
        padding = params['padding']
        
        # 验证输入形状
        if len(input_shape) != 3:
            raise DataValidationError(
                "conv2d_input_shape",
                input_shape,
                "长度为3的元组(C,H,W)"
            )
        
        C_in, H_in, W_in = input_shape
        
        # 计算输出尺寸
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        
        # 计算参数量
        weight_params = out_channels * in_channels * kernel_size * kernel_size
        bias_params = out_channels  # 假设使用偏置
        total_params = weight_params + bias_params
        
        # 计算FLOPs
        flops_per_output = kernel_size * kernel_size * in_channels
        total_flops = 2 * flops_per_output * out_channels * H_out * W_out
        
        # 计算内存占用 (FP32)
        param_memory = total_params * 4 / (1024**2)  # MB
        output_memory = out_channels * H_out * W_out * 4 / (1024**2)  # MB
        total_memory = param_memory + output_memory
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'memory_mb': total_memory,
            'output_shape': (out_channels, H_out, W_out)
        }
    
    def _analyze_linear(
        self, 
        params: Dict[str, Any], 
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """分析全连接层
        
        Args:
            params: 全连接层参数
            input_shape: 输入形状
            
        Returns:
            全连接层分析结果
        """
        in_features = params['in_features']
        out_features = params['out_features']
        
        # 计算输入特征数（假设输入已经展平）
        if len(input_shape) > 1:
            input_features = np.prod(input_shape)
        else:
            input_features = input_shape[0]
        
        # 验证特征数匹配
        if input_features != in_features:
            raise InvalidLayerConfigError(
                layer_type='linear',
                param_name='in_features',
                param_value=in_features,
                expected_type=f'匹配输入{input_features}'
            )
        
        # 计算参数量
        weight_params = in_features * out_features
        bias_params = out_features
        total_params = weight_params + bias_params
        
        # 计算FLOPs
        total_flops = 2 * weight_params
        
        # 计算内存占用
        param_memory = total_params * 4 / (1024**2)  # MB
        output_memory = out_features * 4 / (1024**2)  # MB
        total_memory = param_memory + output_memory
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'memory_mb': total_memory,
            'output_shape': (out_features,)
        }
    
    def _analyze_maxpool2d(
        self, 
        params: Dict[str, Any], 
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """分析最大池化层
        
        Args:
            params: 池化层参数
            input_shape: 输入形状
            
        Returns:
            池化层分析结果
        """
        kernel_size = params['kernel_size']
        stride = params.get('stride', kernel_size)
        padding = params.get('padding', 0)
        
        C_in, H_in, W_in = input_shape
        
        # 计算输出尺寸
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        
        # 池化层没有参数
        total_params = 0
        total_flops = C_in * H_out * W_out  # 比较操作
        
        # 计算内存占用
        output_memory = C_in * H_out * W_out * 4 / (1024**2)  # MB
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'memory_mb': output_memory,
            'output_shape': (C_in, H_out, W_out)
        }