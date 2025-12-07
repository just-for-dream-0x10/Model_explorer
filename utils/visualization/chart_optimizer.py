"""
图表缓存优化器
提供图表渲染性能优化功能
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Union
import plotly.graph_objects as go
import streamlit as st
from utils.cache import CacheManager


class ChartCache:
    """图表缓存管理器"""
    
    def __init__(self, ttl: int = 300):  # 5分钟缓存
        """初始化图表缓存
        
        Args:
            ttl: 缓存生存时间（秒）
        """
        self.cache = CacheManager(default_ttl=ttl)
        self.render_times = {}
    
    def _generate_cache_key(self, chart_data: Dict[str, Any]) -> str:
        """生成图表缓存键
        
        Args:
            chart_data: 图表数据字典
            
        Returns:
            缓存键字符串
        """
        # 序列化数据并生成哈希
        serialized = json.dumps(chart_data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def get_cached_chart(self, chart_data: Dict[str, Any]) -> Optional[go.Figure]:
        """获取缓存的图表
        
        Args:
            chart_data: 图表数据字典
            
        Returns:
            缓存的图表对象或None
        """
        cache_key = self._generate_cache_key(chart_data)
        return self.cache.get(cache_key)
    
    def cache_chart(self, chart_data: Dict[str, Any], fig: go.Figure) -> None:
        """缓存图表
        
        Args:
            chart_data: 图表数据字典
            fig: 图表对象
        """
        cache_key = self._generate_cache_key(chart_data)
        self.cache.set(cache_key, fig)
    
    def render_chart(
        self,
        fig: go.Figure,
        use_container_width: bool = True,
        width: Optional[str] = None,
        key: Optional[str] = None,
        cache_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """渲染图表并记录性能
        
        Args:
            fig: 图表对象
            use_container_width: 是否使用容器宽度
            width: 图表宽度
            key: 图表键
            cache_data: 用于缓存的数据
            
        Returns:
            性能统计信息
        """
        start_time = time.time()
        
        # 渲染图表
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            width=width,
            key=key
        )
        
        render_time = time.time() - start_time
        
        # 记录性能
        stats = {
            "render_time": render_time,
            "timestamp": time.time(),
            "chart_type": type(fig.data[0]).__name__ if fig.data else "unknown"
        }
        
        if key:
            self.render_times[key] = stats
        
        return stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if not self.render_times:
            return {"message": "暂无性能数据"}
        
        times = [stats["render_time"] for stats in self.render_times.values()]
        
        return {
            "total_charts": len(times),
            "avg_render_time": sum(times) / len(times),
            "max_render_time": max(times),
            "min_render_time": min(times),
            "cache_stats": self.cache.get_stats()
        }


class ChartOptimizer:
    """图表优化器"""
    
    @staticmethod
    def optimize_data_density(data: list, max_points: int = 1000) -> list:
        """优化数据密度
        
        Args:
            data: 原始数据列表
            max_points: 最大数据点数
            
        Returns:
            优化后的数据列表
        """
        if len(data) <= max_points:
            return data
        
        # 等间隔采样
        step = len(data) // max_points
        if step > 1:
            return data[::step]
        
        return data
    
    @staticmethod
    def optimize_figure(fig: go.Figure, enable_animations: bool = False) -> go.Figure:
        """优化图表配置
        
        Args:
            fig: 原始图表
            enable_animations: 是否启用动画
            
        Returns:
            优化后的图表
        """
        # 禁用动画以提高性能
        if not enable_animations:
            fig.update_layout(
                transition_duration=0,
                hovermode='x' if len(fig.data) > 1 else 'closest'
            )
        
        # 优化图表配置
        fig.update_layout(
            template='plotly_white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            showlegend=len(fig.data) <= 5  # 数据系列过多时隐藏图例
        )
        
        return fig
    
    @staticmethod
    def create_lightweight_scatter(
        x_data: list,
        y_data: list,
        **kwargs
    ) -> go.Figure:
        """创建轻量级散点图
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            **kwargs: 其他参数
            
        Returns:
            优化的散点图
        """
        # 限制数据点数量
        max_points = kwargs.pop('max_points', 2000)
        x_data = ChartOptimizer.optimize_data_density(x_data, max_points)
        y_data = ChartOptimizer.optimize_data_density(y_data, max_points)
        
        # 创建简化的散点图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=4,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            **kwargs
        ))
        
        # 应用优化配置
        return ChartOptimizer.optimize_figure(fig)


# 全局图表缓存实例
chart_cache = ChartCache()


def cached_plotly_chart(
    fig: go.Figure,
    cache_key_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """带缓存的图表渲染函数
    
    Args:
        fig: 图表对象
        cache_key_data: 用于生成缓存键的数据
        **kwargs: st.plotly_chart的其他参数
    """
    # 如果提供了缓存数据，尝试从缓存获取
    if cache_key_data:
        cached_fig = chart_cache.get_cached_chart(cache_key_data)
        if cached_fig:
            fig = cached_fig
        else:
            # 缓存新图表
            chart_cache.cache_chart(cache_key_data, fig)
    
    # 渲染图表
    chart_cache.render_chart(fig, **kwargs)


def show_chart_performance():
    """显示图表性能统计"""
    stats = chart_cache.get_performance_stats()
    
    if "message" in stats:
        st.info(stats["message"])
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总图表数", stats["total_charts"])
    
    with col2:
        st.metric("平均渲染时间", f"{stats['avg_render_time']:.3f}s")
    
    with col3:
        st.metric("缓存命中率", f"{stats['cache_stats']['hit_rate']:.1%}")
    
    # 显示详细统计
    with st.expander("详细性能数据"):
        st.json(stats)