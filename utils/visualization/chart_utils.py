"""图表工具类

提供统一的图表创建接口，减少重复代码。

Author: Just For Dream Lab
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class ChartBuilder:
    """图表构建器
    
    提供统一的图表创建接口，支持多种图表类型。
    所有图表使用一致的样式和配置。
    
    Example:
        >>> chart = ChartBuilder()
        >>> fig = chart.create_line_chart(
        ...     x_data=[1, 2, 3], 
        ...     y_data=[[1, 2, 3], [2, 4, 6]],
        ...     title="测试图表",
        ...     line_names=["系列1", "系列2"]
        ... )
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    
    # 默认样式配置
    DEFAULT_COLORS = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
    ]
    
    DEFAULT_LAYOUT = {
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
        'plot_bgcolor': 'rgba(240, 240, 240, 0.8)',
        'paper_bgcolor': 'white',
        'showlegend': True,
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
    }
    
    def __init__(self):
        """初始化图表构建器"""
        self.colors = self.DEFAULT_COLORS.copy()
        self.layout_config = self.DEFAULT_LAYOUT.copy()
    
    def create_line_chart(
        self,
        x_data: List[Union[int, float]],
        y_data: Union[List[Union[int, float]], List[List[Union[int, float]]]],
        title: str,
        x_title: str = "X轴",
        y_title: str = "Y轴",
        line_names: Optional[List[str]] = None,
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """创建折线图
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据，可以是单条线或多条线
            title: 图表标题
            x_title: X轴标题
            y_title: Y轴标题
            line_names: 线条名称列表
            height: 图表高度
            **kwargs: 其他参数
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        # 处理单条线的情况
        if not isinstance(y_data[0], list):
            y_data = [y_data]
        
        # 添加每条线
        for i, y_values in enumerate(y_data):
            line_name = line_names[i] if line_names and i < len(line_names) else f"系列{i+1}"
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_values,
                    mode='lines+markers',
                    name=line_name,
                    line={'color': color, 'width': 2},
                    marker={'size': 6}
                )
            )
        
        # 更新布局
        layout = {
            'title': {'text': title, 'x': 0.5},
            'xaxis': {'title': x_title},
            'yaxis': {'title': y_title},
            'height': height,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def create_bar_chart(
        self,
        x_data: List[str],
        y_data: List[Union[int, float]],
        title: str,
        x_title: str = "类别",
        y_title: str = "数值",
        color: Optional[str] = None,
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """创建柱状图
        
        Args:
            x_data: X轴类别数据
            y_data: Y轴数值数据
            title: 图表标题
            x_title: X轴标题
            y_title: Y轴标题
            color: 柱子颜色
            height: 图表高度
            **kwargs: 其他参数
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=y_data,
                marker_color=color or self.colors[0],
                text=[f'{val:.2f}' for val in y_data],
                textposition='auto'
            )
        )
        
        # 更新布局
        layout = {
            'title': {'text': title, 'x': 0.5},
            'xaxis': {'title': x_title},
            'yaxis': {'title': y_title},
            'height': height,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def create_heatmap(
        self,
        data: np.ndarray,
        title: str,
        x_title: str = "X轴",
        y_title: str = "Y轴",
        colorscale: str = "Viridis",
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """创建热力图
        
        Args:
            data: 2D数据数组
            title: 图表标题
            x_title: X轴标题
            y_title: Y轴标题
            colorscale: 颜色映射
            height: 图表高度
            **kwargs: 其他参数
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                colorscale=colorscale,
                showscale=True
            )
        )
        
        # 更新布局
        layout = {
            'title': {'text': title, 'x': 0.5},
            'xaxis': {'title': x_title},
            'yaxis': {'title': y_title},
            'height': height,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def create_pie_chart(
        self,
        labels: List[str],
        values: List[Union[int, float]],
        title: str,
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """创建饼图
        
        Args:
            labels: 标签列表
            values: 数值列表
            title: 图表标题
            height: 图表高度
            **kwargs: 其他参数
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure(
            data=go.Pie(
                labels=labels,
                values=values,
                hole=0.3,  # 甜甜圈图
                textinfo='label+percent',
                textposition='outside'
            )
        )
        
        # 更新布局
        layout = {
            'title': {'text': title, 'x': 0.5},
            'height': height,
            'showlegend': True,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def create_scatter_plot(
        self,
        x_data: List[Union[int, float]],
        y_data: List[Union[int, float]],
        title: str,
        x_title: str = "X轴",
        y_title: str = "Y轴",
        color: Optional[str] = None,
        height: int = 400,
        **kwargs
    ) -> go.Figure:
        """创建散点图
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            title: 图表标题
            x_title: X轴标题
            y_title: Y轴标题
            color: 点的颜色
            height: 图表高度
            **kwargs: 其他参数
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker={
                    'size': 8,
                    'color': color or self.colors[0],
                    'opacity': 0.7
                }
            )
        )
        
        # 更新布局
        layout = {
            'title': {'text': title, 'x': 0.5},
            'xaxis': {'title': x_title},
            'yaxis': {'title': y_title},
            'height': height,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def create_subplots(
        self,
        rows: int,
        cols: int,
        subplot_titles: List[str],
        height: int = 600,
        **kwargs
    ) -> Tuple[go.Figure, List[go.Figure]]:
        """创建子图
        
        Args:
            rows: 行数
            cols: 列数
            subplot_titles: 子图标题列表
            height: 总高度
            **kwargs: 其他参数
            
        Returns:
            (整体图表, 子图列表)
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1 / rows,
            horizontal_spacing=0.1 / cols
        )
        
        # 更新布局
        layout = {
            'height': height,
            **self.layout_config
        }
        layout.update(kwargs)
        fig.update_layout(**layout)
        
        return fig
    
    def display_chart(
        self,
        fig: go.Figure,
        use_container_width: bool = True,
        **kwargs
    ) -> None:
        """显示图表
        
        Args:
            fig: Plotly图表对象
            use_container_width: 是否使用容器宽度
            **kwargs: 其他参数
        """
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            **kwargs
        )


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