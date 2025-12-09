"""
数值稳定性检测器 - Numerical Stability Checker

为所有模块提供统一的稳定性检测接口
遵循项目定位：不仅展示"算了什么"，更要检测"什么时候会出问题"

Author: Neural Network Math Explorer
Date: 2024-01-XX
"""

import numpy as np
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional


class StabilityChecker:
    """
    数值稳定性检查器

    提供统一的检测接口，包括：
    - 梯度检测（消失/爆炸）
    - 激活值检测（过大/过小）
    - 门控饱和检测（LSTM/GRU）
    - 数值验证（梯度正确性）
    - NaN/Inf检测
    """

    # 阈值定义（基于稳定性诊断模块的最佳实践）
    THRESHOLDS = {
        "gradient_vanishing": 1e-7,  # 梯度消失阈值
        "gradient_exploding": 10,  # 梯度爆炸阈值
        "activation_extreme": 100,  # 激活值过大阈值
        "gate_saturation": 0.95,  # 门控饱和阈值
        "numerical_diff_good": 1e-7,  # 梯度验证-优秀
        "numerical_diff_ok": 1e-5,  # 梯度验证-可接受
        "param_exploding": 1e6,  # 参数爆炸阈值
        "learning_rate_high": 1.0,  # 学习率过高阈值
    }

    @staticmethod
    def check_gradient(gradients: np.ndarray, name: str = "梯度") -> Dict[str, Any]:
        """
        检查梯度是否正常

        Args:
            gradients: 梯度数组
            name: 梯度名称（用于显示）

        Returns:
            检测结果字典
        """
        grad_norm = np.linalg.norm(gradients)
        grad_mean = np.mean(np.abs(gradients))
        grad_max = np.max(np.abs(gradients))

        if grad_norm < StabilityChecker.THRESHOLDS["gradient_vanishing"]:
            return {
                "status": "error",
                "type": f"{name}消失",
                "value": f"{grad_norm:.2e}",
                "threshold": f'< {StabilityChecker.THRESHOLDS["gradient_vanishing"]:.0e}',
                "icon": "🔴",
                "severity": "high",
                "details": {
                    "范数": f"{grad_norm:.2e}",
                    "平均绝对值": f"{grad_mean:.2e}",
                    "最大绝对值": f"{grad_max:.2e}",
                },
                "solution": [
                    "使用ResNet残差连接",
                    "使用ReLU激活函数",
                    "使用He初始化",
                    "增加学习率",
                    "检查是否有激活函数饱和",
                ],
                "explanation": "梯度范数过小，反向传播信号几乎消失，导致网络无法学习",
            }
        elif grad_norm > StabilityChecker.THRESHOLDS["gradient_exploding"]:
            return {
                "status": "error",
                "type": f"{name}爆炸",
                "value": f"{grad_norm:.2e}",
                "threshold": f'> {StabilityChecker.THRESHOLDS["gradient_exploding"]}',
                "icon": "🟠",
                "severity": "high",
                "details": {
                    "范数": f"{grad_norm:.2e}",
                    "平均绝对值": f"{grad_mean:.2e}",
                    "最大绝对值": f"{grad_max:.2e}",
                },
                "solution": [
                    "使用梯度裁剪 (gradient clipping)",
                    "降低学习率 (例如从0.01降到0.001)",
                    "检查权重初始化是否过大",
                    "使用BatchNorm/LayerNorm",
                    "检查输入数据是否已归一化",
                ],
                "explanation": "梯度范数过大，参数更新步长太大，可能导致训练不稳定或NaN",
            }
        else:
            return {
                "status": "success",
                "type": f"{name}正常",
                "value": f"{grad_norm:.2e}",
                "icon": "🟢",
                "severity": "none",
                "details": {
                    "范数": f"{grad_norm:.2e}",
                    "平均绝对值": f"{grad_mean:.2e}",
                    "最大绝对值": f"{grad_max:.2e}",
                },
            }

    @staticmethod
    def check_activation(
        activations: np.ndarray, name: str = "激活值"
    ) -> Dict[str, Any]:
        """
        检查激活值是否正常

        Args:
            activations: 激活值数组
            name: 激活值名称

        Returns:
            检测结果字典
        """
        max_val = np.max(np.abs(activations))
        mean_val = np.mean(activations)
        std_val = np.std(activations)

        # 检查NaN或Inf
        if np.isnan(activations).any():
            return {
                "status": "error",
                "type": f"{name}包含NaN",
                "value": f"{np.sum(np.isnan(activations))}个NaN",
                "icon": "🟣",
                "severity": "critical",
                "details": {
                    "NaN数量": np.sum(np.isnan(activations)),
                    "总元素": activations.size,
                    "占比": f"{np.sum(np.isnan(activations))/activations.size*100:.1f}%",
                },
                "solution": [
                    "检查除零错误",
                    "检查log(0)或sqrt(负数)等非法操作",
                    "检查是否有梯度爆炸",
                    "降低学习率",
                    "使用梯度裁剪",
                ],
                "explanation": "NaN (Not a Number) 表示计算出现非法操作，必须立即修复",
            }

        if np.isinf(activations).any():
            return {
                "status": "error",
                "type": f"{name}包含Inf",
                "value": f"{np.sum(np.isinf(activations))}个Inf",
                "icon": "🟣",
                "severity": "critical",
                "details": {
                    "Inf数量": np.sum(np.isinf(activations)),
                    "总元素": activations.size,
                    "占比": f"{np.sum(np.isinf(activations))/activations.size*100:.1f}%",
                },
                "solution": [
                    "检查数值溢出",
                    "检查指数运算",
                    "降低学习率",
                    "使用梯度裁剪",
                    "检查权重初始化",
                ],
                "explanation": "Inf (Infinity) 表示数值溢出，计算结果超出浮点数表示范围",
            }

        if max_val > StabilityChecker.THRESHOLDS["activation_extreme"]:
            return {
                "status": "warning",
                "type": f"{name}过大",
                "value": f"{max_val:.2f}",
                "threshold": f'> {StabilityChecker.THRESHOLDS["activation_extreme"]}',
                "icon": "🟡",
                "severity": "medium",
                "details": {
                    "最大绝对值": f"{max_val:.2f}",
                    "均值": f"{mean_val:.4f}",
                    "标准差": f"{std_val:.4f}",
                },
                "solution": [
                    "使用BatchNorm或LayerNorm",
                    "使用Xavier或He初始化",
                    "检查输入数据范围",
                    "使用激活函数约束输出范围",
                ],
                "explanation": "激活值过大可能导致数值不稳定，增加溢出风险",
            }
        else:
            return {
                "status": "success",
                "type": f"{name}正常",
                "value": f"最大={max_val:.2f}",
                "icon": "🟢",
                "severity": "none",
                "details": {
                    "最大绝对值": f"{max_val:.2f}",
                    "均值": f"{mean_val:.4f}",
                    "标准差": f"{std_val:.4f}",
                },
            }

    @staticmethod
    def check_gate_saturation(
        gate_values: np.ndarray, gate_name: str = "门控"
    ) -> Dict[str, Any]:
        """
        检查门控是否饱和（用于LSTM/GRU）

        Args:
            gate_values: 门控值数组（应该在0-1之间）
            gate_name: 门控名称（如"遗忘门"、"更新门"）

        Returns:
            检测结果字典
        """
        # 计算饱和率（接近0或1的比例）
        near_zero = np.sum(gate_values < 0.05)
        near_one = np.sum(gate_values > 0.95)
        total = gate_values.size
        saturation_rate = (near_zero + near_one) / total

        mean_val = np.mean(gate_values)
        std_val = np.std(gate_values)

        if saturation_rate > StabilityChecker.THRESHOLDS["gate_saturation"]:
            return {
                "status": "warning",
                "type": f"{gate_name}饱和",
                "value": f"{saturation_rate*100:.1f}%",
                "threshold": f'> {StabilityChecker.THRESHOLDS["gate_saturation"]*100:.0f}%',
                "icon": "🟡",
                "severity": "medium",
                "details": {
                    "饱和率": f"{saturation_rate*100:.1f}%",
                    "接近0": f"{near_zero}/{total} ({near_zero/total*100:.1f}%)",
                    "接近1": f"{near_one}/{total} ({near_one/total*100:.1f}%)",
                    "均值": f"{mean_val:.4f}",
                    "标准差": f"{std_val:.4f}",
                },
                "solution": [
                    "降低学习率",
                    "使用BatchNorm/LayerNorm",
                    "检查权重初始化（使用Orthogonal初始化）",
                    "使用梯度裁剪",
                    "考虑使用更小的网络",
                ],
                "explanation": "门控值过度集中在0或1，导致信息流动受阻，类似梯度消失",
            }
        else:
            return {
                "status": "success",
                "type": f"{gate_name}正常",
                "value": f"饱和率={saturation_rate*100:.1f}%",
                "icon": "🟢",
                "severity": "none",
                "details": {
                    "饱和率": f"{saturation_rate*100:.1f}%",
                    "接近0": f"{near_zero}/{total} ({near_zero/total*100:.1f}%)",
                    "接近1": f"{near_one}/{total} ({near_one/total*100:.1f}%)",
                    "均值": f"{mean_val:.4f}",
                    "标准差": f"{std_val:.4f}",
                },
            }

    @staticmethod
    def verify_gradient(
        numerical_grad: np.ndarray, analytical_grad: np.ndarray, name: str = "梯度"
    ) -> Dict[str, Any]:
        """
        验证梯度计算正确性（数值梯度 vs 解析梯度）

        参考反向传播模块的标准（第318-324行）

        Args:
            numerical_grad: 数值梯度（有限差分法计算）
            analytical_grad: 解析梯度（反向传播计算）
            name: 梯度名称

        Returns:
            检测结果字典
        """
        diff = np.abs(numerical_grad - analytical_grad).mean()
        relative_error = diff / (
            np.abs(numerical_grad).mean() + np.abs(analytical_grad).mean() + 1e-8
        )

        if diff < StabilityChecker.THRESHOLDS["numerical_diff_good"]:
            return {
                "status": "success",
                "type": f"{name}验证",
                "value": f"{diff:.2e}",
                "threshold": f'< {StabilityChecker.THRESHOLDS["numerical_diff_good"]:.0e}',
                "message": "✅ 梯度计算正确",
                "icon": "✅",
                "severity": "none",
                "details": {
                    "平均差异": f"{diff:.2e}",
                    "相对误差": f"{relative_error:.2e}",
                    "数值梯度范数": f"{np.linalg.norm(numerical_grad):.2e}",
                    "解析梯度范数": f"{np.linalg.norm(analytical_grad):.2e}",
                },
            }
        elif diff < StabilityChecker.THRESHOLDS["numerical_diff_ok"]:
            return {
                "status": "warning",
                "type": f"{name}验证",
                "value": f"{diff:.2e}",
                "threshold": f'< {StabilityChecker.THRESHOLDS["numerical_diff_ok"]:.0e}',
                "message": "⚠️ 可能有小误差",
                "icon": "⚠️",
                "severity": "low",
                "details": {
                    "平均差异": f"{diff:.2e}",
                    "相对误差": f"{relative_error:.2e}",
                    "数值梯度范数": f"{np.linalg.norm(numerical_grad):.2e}",
                    "解析梯度范数": f"{np.linalg.norm(analytical_grad):.2e}",
                },
                "solution": [
                    "检查链式法则是否正确",
                    "检查激活函数导数",
                    "增加数值梯度的精度（减小epsilon）",
                ],
                "explanation": "误差在可接受范围内，但可以进一步优化",
            }
        else:
            return {
                "status": "error",
                "type": f"{name}验证",
                "value": f"{diff:.2e}",
                "threshold": f'> {StabilityChecker.THRESHOLDS["numerical_diff_ok"]:.0e}',
                "message": "❌ 梯度计算可能有误",
                "icon": "❌",
                "severity": "high",
                "details": {
                    "平均差异": f"{diff:.2e}",
                    "相对误差": f"{relative_error:.2e}",
                    "数值梯度范数": f"{np.linalg.norm(numerical_grad):.2e}",
                    "解析梯度范数": f"{np.linalg.norm(analytical_grad):.2e}",
                },
                "solution": [
                    "仔细检查反向传播实现",
                    "逐步验证每个梯度计算",
                    "检查链式法则是否正确应用",
                    "检查矩阵维度是否匹配",
                    "参考反向传播模块的实现",
                ],
                "explanation": "数值梯度和解析梯度差异过大，反向传播实现可能有错误",
            }

    @staticmethod
    def check_learning_rate(
        learning_rate: float, grad_norm: float, param_norm: float
    ) -> Dict[str, Any]:
        """
        检查学习率是否合适

        Args:
            learning_rate: 学习率
            grad_norm: 梯度范数
            param_norm: 参数范数

        Returns:
            检测结果字典
        """
        # 估计参数更新的步长
        update_norm = learning_rate * grad_norm
        relative_update = update_norm / (param_norm + 1e-8)

        if learning_rate > StabilityChecker.THRESHOLDS["learning_rate_high"]:
            return {
                "status": "warning",
                "type": "学习率过高",
                "value": f"{learning_rate}",
                "threshold": f'> {StabilityChecker.THRESHOLDS["learning_rate_high"]}',
                "icon": "🟡",
                "severity": "medium",
                "details": {
                    "学习率": f"{learning_rate}",
                    "预估更新步长": f"{update_norm:.2e}",
                    "相对更新比例": f"{relative_update:.2%}",
                },
                "solution": [
                    "降低学习率（建议<0.1）",
                    "使用学习率衰减",
                    "使用自适应学习率（Adam, RMSprop）",
                ],
                "explanation": "学习率过高可能导致训练不稳定或发散",
            }
        elif relative_update > 0.1:
            return {
                "status": "warning",
                "type": "参数更新过大",
                "value": f"{relative_update:.2%}",
                "threshold": "> 10%",
                "icon": "🟡",
                "severity": "medium",
                "details": {
                    "学习率": f"{learning_rate}",
                    "梯度范数": f"{grad_norm:.2e}",
                    "参数范数": f"{param_norm:.2e}",
                    "相对更新比例": f"{relative_update:.2%}",
                },
                "solution": ["降低学习率", "使用梯度裁剪", "使用权重衰减"],
                "explanation": "单步更新超过参数大小的10%，可能导致训练不稳定",
            }
        else:
            return {
                "status": "success",
                "type": "学习率合适",
                "value": f"{learning_rate}",
                "icon": "🟢",
                "severity": "none",
                "details": {
                    "学习率": f"{learning_rate}",
                    "预估更新步长": f"{update_norm:.2e}",
                    "相对更新比例": f"{relative_update:.2%}",
                },
            }

    @staticmethod
    def display_issues(
        issues: List[Dict[str, Any]], title: str = "🔬 数值稳定性诊断报告"
    ):
        """
        在Streamlit中显示检测结果

        参考稳定性诊断模块的显示方式

        Args:
            issues: 检测结果列表
            title: 报告标题
        """
        if not issues:
            st.success("✅ 所有检查通过，没有发现问题")
            return

        # 分组
        critical = [i for i in issues if i.get("severity") == "critical"]
        errors = [
            i
            for i in issues
            if i["status"] == "error" and i.get("severity") != "critical"
        ]
        warnings = [i for i in issues if i["status"] == "warning"]
        success = [i for i in issues if i["status"] == "success"]

        # 显示标题
        st.markdown(f"### {title}")

        # 关键问题（必须立即修复）
        if critical:
            st.error("🚨 **关键问题（必须立即修复）**")
            for issue in critical:
                with st.expander(
                    f"{issue['icon']} {issue['type']}: {issue['value']}", expanded=True
                ):
                    st.write(f"**说明**: {issue.get('explanation', '')}")

                    if "details" in issue:
                        st.write("**详细信息**:")
                        for key, val in issue["details"].items():
                            st.write(f"- {key}: `{val}`")

                    if "solution" in issue:
                        st.write("**🔧 解决方案**:")
                        for i, sol in enumerate(issue["solution"], 1):
                            st.write(f"{i}. {sol}")

        # 错误（需要修复）
        if errors:
            st.error("❌ **检测到问题（需要修复）**")

            # 创建问题表格
            table_data = []
            for issue in errors:
                table_data.append(
                    {
                        "状态": issue["icon"],
                        "问题类型": issue["type"],
                        "当前值": issue["value"],
                        "阈值": issue.get("threshold", "N/A"),
                    }
                )

            df = pd.DataFrame(table_data)
            st.markdown(df.to_markdown(index=False))

            # 详细信息
            for issue in errors:
                with st.expander(f"详情: {issue['type']}", expanded=False):
                    st.write(f"**说明**: {issue.get('explanation', '')}")

                    if "details" in issue:
                        st.write("**详细信息**:")
                        for key, val in issue["details"].items():
                            st.write(f"- {key}: `{val}`")

                    if "solution" in issue:
                        st.write("**🔧 解决方案**:")
                        for i, sol in enumerate(issue["solution"], 1):
                            st.write(f"{i}. {sol}")

        # 警告（建议优化）
        if warnings:
            st.warning("⚠️ **警告（建议优化）**")

            table_data = []
            for issue in warnings:
                table_data.append(
                    {
                        "状态": issue["icon"],
                        "问题类型": issue["type"],
                        "当前值": issue["value"],
                        "阈值": issue.get("threshold", "N/A"),
                    }
                )

            df = pd.DataFrame(table_data)
            st.markdown(df.to_markdown(index=False))

            # 详细信息
            for issue in warnings:
                with st.expander(f"详情: {issue['type']}", expanded=False):
                    st.write(f"**说明**: {issue.get('explanation', '')}")

                    if "details" in issue:
                        st.write("**详细信息**:")
                        for key, val in issue["details"].items():
                            st.write(f"- {key}: `{val}`")

                    if "solution" in issue:
                        st.write("**💡 建议**:")
                        for i, sol in enumerate(issue["solution"], 1):
                            st.write(f"{i}. {sol}")

        # 成功的检查
        if success:
            with st.expander("✅ 通过的检查", expanded=False):
                for issue in success:
                    st.success(f"{issue['icon']} {issue['type']}: {issue['value']}")
                    if "details" in issue:
                        for key, val in issue["details"].items():
                            st.write(f"- {key}: `{val}`")


def compute_numerical_gradient(neuron, input_data, upstream_gradient, epsilon=1e-5):
    """
    计算数值梯度（有限差分法）

    用于验证解析梯度的正确性

    Args:
        neuron: 神经元对象
        input_data: 输入数据
        upstream_gradient: 上游梯度
        epsilon: 扰动大小

    Returns:
        数值梯度
    """
    numerical_grads = np.zeros_like(neuron.weights)

    for i in range(neuron.weights.size):
        # 扁平化索引
        idx = (
            np.unravel_index(i, neuron.weights.shape) if neuron.weights.ndim > 1 else i
        )

        # 前向扰动
        original = (
            neuron.weights.flat[i] if neuron.weights.ndim > 1 else neuron.weights[i]
        )
        neuron.weights.flat[i] = original + epsilon
        output_plus = neuron.forward(input_data)
        loss_plus = np.sum(output_plus * upstream_gradient)

        # 后向扰动
        neuron.weights.flat[i] = original - epsilon
        output_minus = neuron.forward(input_data)
        loss_minus = np.sum(output_minus * upstream_gradient)

        # 恢复原值
        neuron.weights.flat[i] = original

        # 计算数值梯度
        numerical_grads.flat[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return numerical_grads
