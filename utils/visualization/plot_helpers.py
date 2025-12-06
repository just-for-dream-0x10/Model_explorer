"""绘图辅助函数

提供专门的绘图功能，用于特定场景的可视化需求。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from .chart_utils import ChartBuilder


class NetworkVisualization:
    """网络可视化专用类
    
    提供神经网络相关的可视化功能。
    """
    
    def __init__(self):
        """初始化网络可视化工具"""
        self.chart_builder = ChartBuilder()
    
    def plot_network_architecture(
        self,
        layer_shapes: List[Tuple[int, ...]],
        layer_names: List[str],
        title: str = "网络架构"
    ) -> go.Figure:
        """绘制网络架构图
        
        Args:
            layer_shapes: 各层的形状列表
            layer_names: 各层的名称列表
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        # 计算每层的位置
        x_positions = list(range(len(layer_shapes)))
        y_positions = [0] * len(layer_shapes)
        
        # 计算节点大小（基于参数量）
        node_sizes = []
        for shape in layer_shapes:
            params = np.prod(shape)
            size = min(50, max(10, params / 1000))
            node_sizes.append(size)
        
        # 添加连接线
        for i in range(len(layer_shapes) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i], x_positions[i + 1]],
                    y=[y_positions[i], y_positions[i + 1]],
                    mode='lines',
                    line={'color': 'lightgray', 'width': 2},
                    showlegend=False,
                    hoverinfo='none'
                )
            )
        
        # 添加节点
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker={
                    'size': node_sizes,
                    'color': range(len(layer_shapes)),
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'colorbar': {'title': '参数量'}
                },
                text=layer_names,
                textposition='middle center',
                hovertemplate='<b>%{text}</b><br>形状: %{customdata}<extra></extra>',
                customdata=[str(shape) for shape in layer_shapes]
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            height=300,
            showlegend=False
        )
        
        return fig
    
    def plot_training_curves(
        self,
        curves_data: Dict[str, Dict[str, List[float]]],
        metric: str = "loss"
    ) -> go.Figure:
        """绘制训练曲线
        
        Args:
            curves_data: 模型名 -> 训练数据的字典
            metric: 指标类型 ("loss" 或 "accuracy")
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("训练集", "验证集")
        )
        
        colors = ChartBuilder.DEFAULT_COLORS
        
        for idx, (model_name, curves) in enumerate(curves_data.items()):
            color = colors[idx % len(colors)]
            epochs = curves['epochs']
            
            if metric == "loss":
                train_data = curves['train_loss']
                val_data = curves['val_loss']
                ylabel = "Loss"
            else:
                train_data = curves['train_acc']
                val_data = curves['val_acc']
                ylabel = "Accuracy"
            
            # 训练集曲线
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_data,
                    mode='lines',
                    name=model_name,
                    line={'color': color, 'width': 2},
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 验证集曲线
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_data,
                    mode='lines',
                    name=model_name,
                    line={'color': color, 'width': 2, 'dash': 'dash'},
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=2)
        
        fig.update_layout(
            title=f"{ylabel}曲线对比（实线=训练集，虚线=验证集）",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_parameter_distribution(
        self,
        param_data: Dict[str, List[int]],
        title: str = "参数量分布"
    ) -> go.Figure:
        """绘制参数量分布图
        
        Args:
            param_data: 层名 -> 参数量的字典
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        layers = list(param_data.keys())
        params = list(param_data.values())
        
        fig.add_trace(
            go.Bar(
                x=layers,
                y=params,
                marker_color=ChartBuilder.DEFAULT_COLORS[0],
                text=[self._format_params(p) for p in params],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="网络层",
            yaxis_title="参数量",
            height=400
        )
        
        return fig
    
    def plot_gradient_flow(
        self,
        gradient_data: Dict[str, List[float]],
        title: str = "梯度流分析"
    ) -> go.Figure:
        """绘制梯度流图
        
        Args:
            gradient_data: 层名 -> 梯度值的字典
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        layers = list(gradient_data.keys())
        grads = list(gradient_data.values())
        
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=grads,
                mode='lines+markers',
                line={'color': 'red', 'width': 2},
                marker={'size': 8}
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="网络层",
            yaxis_title="梯度范数",
            yaxis_type="log",
            height=400
        )
        
        return fig
    
    def _format_params(self, num: int) -> str:
        """格式化参数量显示"""
        if num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return str(num)


class MathVisualization:
    """数学可视化专用类
    
    提供数学概念的可视化功能。
    """
    
    @staticmethod
    def plot_convolution_process(
        input_matrix: np.ndarray,
        kernel: np.ndarray,
        output_matrix: np.ndarray,
        title: str = "卷积过程"
    ) -> go.Figure:
        """可视化卷积过程
        
        Args:
            input_matrix: 输入矩阵
            kernel: 卷积核
            output_matrix: 输出矩阵
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("输入", "卷积核", "输出"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # 输入矩阵
        fig.add_trace(
            go.Heatmap(
                z=input_matrix,
                colorscale='Blues',
                showscale=False,
                name="输入"
            ),
            row=1, col=1
        )
        
        # 卷积核
        fig.add_trace(
            go.Heatmap(
                z=kernel,
                colorscale='RdBu',
                showscale=False,
                name="卷积核"
            ),
            row=1, col=2
        )
        
        # 输出矩阵
        fig.add_trace(
            go.Heatmap(
                z=output_matrix,
                colorscale='Viridis',
                showscale=True,
                name="输出"
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title=title,
            height=300
        )
        
        return fig
    
    @staticmethod
    def plot_activation_functions(
        x_range: Tuple[float, float] = (-5, 5),
        num_points: int = 100
    ) -> go.Figure:
        """绘制激活函数对比
        
        Args:
            x_range: x轴范围
            num_points: 采样点数
            
        Returns:
            Plotly图表对象
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # 计算各种激活函数
        activations = {
            'ReLU': np.maximum(0, x),
            'Sigmoid': 1 / (1 + np.exp(-x)),
            'Tanh': np.tanh(x),
            'Leaky ReLU': np.maximum(0.01 * x, x)
        }
        
        fig = go.Figure()
        
        colors = ChartBuilder.DEFAULT_COLORS
        
        for i, (name, y) in enumerate(activations.items()):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=name,
                    line={'color': colors[i % len(colors)], 'width': 2}
                )
            )
        
        fig.update_layout(
            title="激活函数对比",
            xaxis_title="输入值",
            yaxis_title="输出值",
            height=400,
            hovermode='x unified'
        )
        
        return fig


class PlotHelper:
    """绘图辅助类
    
    提供常用的绘图功能和配置。
    """
    
    @staticmethod
    def get_default_colors() -> List[str]:
        """获取默认颜色列表"""
        return ChartBuilder.DEFAULT_COLORS.copy()
    
    @staticmethod
    def format_number(num: Union[int, float], precision: int = 2) -> str:
        """格式化数字显示
        
        Args:
            num: 数字
            precision: 小数位数
            
        Returns:
            格式化后的字符串
        """
        if abs(num) >= 1e9:
            return f"{num/1e9:.{precision}f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.{precision}f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"
    
    @staticmethod
    def create_comparison_table(
        data: Dict[str, Dict[str, Union[int, float]]],
        title: str
    ) -> None:
        """创建对比表格
        
        Args:
            data: 对比数据
            title: 表格标题
        """
        st.markdown(f"### {title}")
        
        df_data = []
        for key, values in data.items():
            row = {'项目': key}
            row.update(values)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def show_metrics(
        metrics: Dict[str, Union[str, int, float]],
        columns: int = 3
    ) -> None:
        """显示指标卡片
        
        Args:
            metrics: 指标字典
            columns: 列数
        """
        cols = st.columns(columns)
        
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(value, (int, float)):
                    st.metric(key, value)
                else:
                    st.metric(key, value)