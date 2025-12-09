"""内存分析器核心模块

提供神经网络内存占用的分析和计算功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..exceptions import (
    InvalidLayerConfigError,
    InsufficientMemoryError,
    ComputationError,
)


class MemoryAnalyzer:
    """内存分析器

    提供神经网络各层内存占用的详细分析。
    """

    # 内存单位转换常量
    BYTES_PER_KB = 1024
    BYTES_PER_MB = 1024**2
    BYTES_PER_GB = 1024**3

    # 数据类型字节数
    DTYPE_BYTES = {
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
    }

    def __init__(self, dtype: str = "float32"):
        """初始化内存分析器

        Args:
            dtype: 数据类型，默认为float32

        Raises:
            InvalidLayerConfigError: 当数据类型无效时
        """
        if dtype not in self.DTYPE_BYTES:
            raise InvalidLayerConfigError(
                layer_type="MemoryAnalyzer",
                param_name="dtype",
                param_value=dtype,
                expected_type=f"支持的数据类型: {list(self.DTYPE_BYTES.keys())}",
            )

        self.dtype = dtype
        self.bytes_per_element = self.DTYPE_BYTES[dtype]

    def analyze_layer_memory(
        self,
        layer_type: str,
        params: Dict[str, Any],
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """分析单个层的内存占用

        Args:
            layer_type: 层类型
            params: 层参数
            input_shape: 输入形状

        Returns:
            内存分析结果

        Raises:
            InvalidLayerConfigError: 当参数无效时
        """
        if layer_type == "conv2d":
            return self._analyze_conv2d_memory(params, input_shape)
        elif layer_type == "linear":
            return self._analyze_linear_memory(params, input_shape)
        elif layer_type == "maxpool2d":
            return self._analyze_maxpool2d_memory(params, input_shape)
        elif layer_type == "batchnorm2d":
            return self._analyze_batchnorm2d_memory(params)
        elif layer_type == "dropout":
            return self._analyze_dropout_memory(params, input_shape)
        else:
            # 默认处理：只计算激活值内存
            return {
                "layer_type": layer_type,
                "parameter_memory_mb": 0.0,
                "activation_memory_mb": self._calculate_activation_memory(input_shape),
                "gradient_memory_mb": self._calculate_gradient_memory(input_shape),
                "total_memory_mb": self._calculate_activation_memory(input_shape),
                "details": {"note": "使用默认内存计算"},
            }

    def analyze_network_memory(
        self,
        layers: List[Dict[str, Any]],
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """分析整个网络的内存占用

        Args:
            layers: 层配置列表
            input_shape: 输入形状
            batch_size: 批次大小

        Returns:
            网络内存分析结果

        Raises:
            ComputationError: 当分析过程中出现错误时
        """
        try:
            current_shape = (batch_size,) + input_shape
            total_param_memory = 0.0
            total_activation_memory = 0.0
            total_gradient_memory = 0.0
            peak_memory = 0.0

            layer_details = []

            for i, layer in enumerate(layers):
                layer_result = self.analyze_layer_memory(
                    layer["layer_type"], layer["params"], current_shape
                )

                total_param_memory += layer_result["parameter_memory_mb"]
                total_activation_memory += layer_result["activation_memory_mb"]
                total_gradient_memory += layer_result["gradient_memory_mb"]

                # 更新当前形状
                if "output_shape" in layer_result:
                    current_shape = (batch_size,) + layer_result["output_shape"]

                # 计算峰值内存（参数 + 当前激活值 + 梯度）
                layer_peak = (
                    total_param_memory
                    + layer_result["activation_memory_mb"]
                    + layer_result["gradient_memory_mb"]
                )
                peak_memory = max(peak_memory, layer_peak)

                layer_details.append(
                    {
                        "layer_index": i,
                        "layer_type": layer["layer_type"],
                        **layer_result,
                    }
                )

            return {
                "total_parameter_memory_mb": total_param_memory,
                "total_activation_memory_mb": total_activation_memory,
                "total_gradient_memory_mb": total_gradient_memory,
                "peak_memory_mb": peak_memory,
                "input_shape": input_shape,
                "batch_size": batch_size,
                "dtype": self.dtype,
                "layer_details": layer_details,
                "summary": self._generate_memory_summary(
                    total_param_memory,
                    total_activation_memory,
                    total_gradient_memory,
                    peak_memory,
                ),
            }
        except Exception as e:
            raise ComputationError(
                operation="网络内存分析", error_details=str(e)
            ) from e

    def _analyze_conv2d_memory(
        self, params: Dict[str, Any], input_shape: Optional[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """分析卷积层内存占用"""
        in_channels = params["in_channels"]
        out_channels = params["out_channels"]
        kernel_size = params["kernel_size"]
        use_bias = params.get("use_bias", True)

        # 参数内存
        weight_params = in_channels * out_channels * kernel_size * kernel_size
        bias_params = out_channels if use_bias else 0
        total_params = weight_params + bias_params
        param_memory = total_params * self.bytes_per_element / self.BYTES_PER_MB

        # 激活值内存
        if input_shape:
            N, C_in, H_in, W_in = input_shape
            # 计算输出尺寸
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            output_shape = (N, out_channels, H_out, W_out)
            activation_memory = (
                np.prod(output_shape) * self.bytes_per_element / self.BYTES_PER_MB
            )
            gradient_memory = activation_memory * 2  # 输入梯度 + 输出梯度
        else:
            output_shape = None
            activation_memory = 0.0
            gradient_memory = 0.0

        return {
            "layer_type": "conv2d",
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": activation_memory,
            "gradient_memory_mb": gradient_memory,
            "total_memory_mb": param_memory + activation_memory,
            "output_shape": output_shape[1:] if output_shape else None,
            "details": {
                "weight_parameters": weight_params,
                "bias_parameters": bias_params,
                "total_parameters": total_params,
                "dtype": self.dtype,
            },
        }

    def _analyze_linear_memory(
        self, params: Dict[str, Any], input_shape: Optional[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """分析全连接层内存占用"""
        in_features = params["in_features"]
        out_features = params["out_features"]
        use_bias = params.get("use_bias", True)

        # 参数内存
        weight_params = in_features * out_features
        bias_params = out_features if use_bias else 0
        total_params = weight_params + bias_params
        param_memory = total_params * self.bytes_per_element / self.BYTES_PER_MB

        # 激活值内存
        if input_shape:
            N = input_shape[0]
            output_shape = (N, out_features)
            activation_memory = (
                np.prod(output_shape) * self.bytes_per_element / self.BYTES_PER_MB
            )
            gradient_memory = activation_memory * 2
        else:
            output_shape = None
            activation_memory = 0.0
            gradient_memory = 0.0

        return {
            "layer_type": "linear",
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": activation_memory,
            "gradient_memory_mb": gradient_memory,
            "total_memory_mb": param_memory + activation_memory,
            "output_shape": output_shape[1:] if output_shape else None,
            "details": {
                "weight_parameters": weight_params,
                "bias_parameters": bias_params,
                "total_parameters": total_params,
                "dtype": self.dtype,
            },
        }

    def _analyze_maxpool2d_memory(
        self, params: Dict[str, Any], input_shape: Optional[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """分析最大池化层内存占用"""
        # 池化层没有参数
        param_memory = 0.0

        # 激活值内存（与输入相同）
        if input_shape:
            activation_memory = self._calculate_activation_memory(input_shape)
            gradient_memory = activation_memory * 2
        else:
            activation_memory = 0.0
            gradient_memory = 0.0

        return {
            "layer_type": "maxpool2d",
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": activation_memory,
            "gradient_memory_mb": gradient_memory,
            "total_memory_mb": activation_memory,
            "details": {"note": "池化层无参数，仅占用激活值内存", "dtype": self.dtype},
        }

    def _analyze_batchnorm2d_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """分析批归一化层内存占用"""
        num_features = params["num_features"]

        # 参数内存：weight + bias + running_mean + running_var
        total_params = num_features * 4
        param_memory = total_params * self.bytes_per_element / self.BYTES_PER_MB

        return {
            "layer_type": "batchnorm2d",
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": 0.0,  # BN通常就地操作
            "gradient_memory_mb": param_memory * 2,  # 参数梯度
            "total_memory_mb": param_memory,
            "details": {
                "total_parameters": total_params,
                "per_feature_parameters": 4,
                "dtype": self.dtype,
            },
        }

    def _analyze_dropout_memory(
        self, params: Dict[str, Any], input_shape: Optional[Tuple[int, ...]]
    ) -> Dict[str, Any]:
        """分析Dropout层内存占用"""
        # Dropout没有参数，但需要存储mask
        param_memory = 0.0

        if input_shape:
            # 激活值内存 + mask内存
            activation_memory = self._calculate_activation_memory(input_shape)
            mask_memory = activation_memory  # mask通常与激活值同样大小
            total_activation_memory = activation_memory + mask_memory
            gradient_memory = activation_memory * 2
        else:
            total_activation_memory = 0.0
            gradient_memory = 0.0

        return {
            "layer_type": "dropout",
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": total_activation_memory,
            "gradient_memory_mb": gradient_memory,
            "total_memory_mb": total_activation_memory,
            "details": {
                "dropout_rate": params.get("p", 0.5),
                "note": "包含mask存储开销",
                "dtype": self.dtype,
            },
        }

    def _calculate_activation_memory(self, shape: Optional[Tuple[int, ...]]) -> float:
        """计算激活值内存占用

        Args:
            shape: 张量形状

        Returns:
            内存占用（MB）
        """
        if not shape:
            return 0.0

        num_elements = np.prod(shape)
        return num_elements * self.bytes_per_element / self.BYTES_PER_MB

    def _calculate_gradient_memory(self, shape: Optional[Tuple[int, ...]]) -> float:
        """计算梯度内存占用

        Args:
            shape: 张量形状

        Returns:
            内存占用（MB）
        """
        # 梯度通常与激活值同样大小，但需要存储输入和输出梯度
        return self._calculate_activation_memory(shape) * 2

    def _generate_memory_summary(
        self,
        param_memory: float,
        activation_memory: float,
        gradient_memory: float,
        peak_memory: float,
    ) -> str:
        """生成内存占用摘要

        Args:
            param_memory: 参数内存
            activation_memory: 激活值内存
            gradient_memory: 梯度内存
            peak_memory: 峰值内存

        Returns:
            摘要字符串
        """
        return (
            f"参数内存: {param_memory:.2f}MB | "
            f"激活值内存: {activation_memory:.2f}MB | "
            f"梯度内存: {gradient_memory:.2f}MB | "
            f"峰值内存: {peak_memory:.2f}MB"
        )

    def check_memory_feasibility(
        self, required_memory_mb: float, available_memory_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """检查内存可行性

        Args:
            required_memory_mb: 所需内存
            available_memory_mb: 可用内存

        Returns:
            可行性检查结果
        """
        if available_memory_mb and required_memory_mb > available_memory_mb:
            raise InsufficientMemoryError(required_memory_mb, available_memory_mb)

        # 内存使用建议
        if required_memory_mb < 100:
            level = "低"
            suggestion = "内存使用较少，可以正常训练"
        elif required_memory_mb < 1000:
            level = "中等"
            suggestion = "建议使用较小的批次大小"
        elif required_memory_mb < 8000:
            level = "高"
            suggestion = "建议使用梯度累积或模型并行"
        else:
            level = "极高"
            suggestion = "必须使用模型并行或分布式训练"

        return {
            "required_memory_mb": required_memory_mb,
            "available_memory_mb": available_memory_mb,
            "usage_level": level,
            "suggestion": suggestion,
            "feasible": available_memory_mb is None
            or required_memory_mb <= available_memory_mb,
        }
