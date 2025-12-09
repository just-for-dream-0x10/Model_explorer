"""性能监控模块

提供缓存使用情况和性能指标的监控功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager
import streamlit as st

from .cache import get_cache_manager


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    cache_stats: Dict[str, Any]
    operation_time: Optional[float] = None
    operation_name: Optional[str] = None


class PerformanceMonitor:
    """性能监控器

    监控系统性能和缓存使用情况。
    """

    def __init__(self, max_history: int = 1000):
        """初始化性能监控器

        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.cache_manager = get_cache_manager()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控

        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            try:
                # 获取系统性能指标
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024**2)

                # 获取缓存统计
                cache_stats = self.cache_manager.get_stats()

                # 记录指标
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    cache_stats=cache_stats,
                )

                self.metrics_history.append(metrics)

                time.sleep(interval)

            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(interval)

    def record_operation(self, operation_name: str, operation_time: float):
        """记录操作时间

        Args:
            operation_name: 操作名称
            operation_time: 操作耗时（秒）
        """
        self.operation_times[operation_name].append(operation_time)

        # 限制每个操作的历史记录数
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name].pop(0)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标

        Returns:
            最新的性能指标，如果没有则返回None
        """
        return self.metrics_history[-1] if self.metrics_history else None

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """获取操作统计信息

        Args:
            operation_name: 操作名称

        Returns:
            操作统计信息字典
        """
        times = self.operation_times.get(operation_name, [])

        if not times:
            return {}

        return {
            "count": len(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "last_time": times[-1],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要

        Returns:
            性能摘要字典
        """
        if not self.metrics_history:
            return {}

        # 计算最近的统计
        recent_metrics = list(self.metrics_history)[-10:]  # 最近10个记录

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)

        # 缓存统计
        cache_stats = self.cache_manager.get_stats()

        # 操作统计
        operation_summary = {}
        for op_name in self.operation_times:
            operation_summary[op_name] = self.get_operation_stats(op_name)

        return {
            "monitoring_duration": time.time() - self.metrics_history[0].timestamp,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "current_cpu": self.metrics_history[-1].cpu_percent,
            "current_memory_mb": self.metrics_history[-1].memory_mb,
            "cache_stats": cache_stats,
            "operation_stats": operation_summary,
            "total_samples": len(self.metrics_history),
        }

    def display_performance_dashboard(self):
        """显示性能监控仪表板"""
        st.markdown("## 📊 性能监控仪表板")

        if not self.metrics_history:
            st.warning("暂无性能数据，请先开始监控")
            return

        summary = self.get_performance_summary()

        # 系统资源监控
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "CPU使用率",
                f"{summary['current_cpu']:.1f}%",
                delta=f"{summary['current_cpu'] - summary['avg_cpu_percent']:.1f}%",
            )

        with col2:
            st.metric(
                "内存使用",
                f"{summary['current_memory_mb']:.1f}MB",
                delta=f"{summary['current_memory_mb'] - summary['avg_memory_mb']:.1f}MB",
            )

        with col3:
            st.metric("监控时长", f"{summary['monitoring_duration']:.0f}秒")

        # 缓存统计
        st.markdown("### 🗄️ 缓存统计")

        cache_stats = summary["cache_stats"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总条目", cache_stats["total_entries"])

        with col2:
            st.metric("有效条目", cache_stats["valid_entries"])

        with col3:
            st.metric("过期条目", cache_stats["expired_entries"])

        with col4:
            st.metric("使用率", f"{cache_stats['usage_ratio']:.1%}")

        # 操作性能统计
        if summary["operation_stats"]:
            st.markdown("### ⚡ 操作性能统计")

            for op_name, stats in summary["operation_stats"].items():
                with st.expander(f"📈 {op_name}", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("调用次数", stats["count"])

                    with col2:
                        st.metric("平均耗时", f"{stats['avg_time']:.3f}s")

                    with col3:
                        st.metric("总耗时", f"{stats['total_time']:.3f}s")

                    # 显示时间趋势
                    if stats["count"] > 1:
                        times = self.operation_times[op_name][-20:]  # 最近20次
                        st.line_chart(
                            list(range(len(times))),
                            times,
                            caption=f"{op_name} 最近20次耗时趋势",
                        )

    def start_operation_timer(self, operation_name: str):
        """开始操作计时

        Args:
            operation_name: 操作名称

        Returns:
            计时器上下文管理器
        """
        return OperationTimer(self, operation_name)


class OperationTimer:
    """操作计时器上下文管理器"""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        """初始化计时器

        Args:
            monitor: 性能监控器实例
            operation_name: 操作名称
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time: Optional[float] = None

    def __enter__(self):
        """进入上下文"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if self.start_time is not None:
            operation_time = time.time() - self.start_time
            self.monitor.record_operation(self.operation_name, operation_time)


# 全局性能监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例

    Returns:
        性能监控器实例
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_performance_monitoring(interval: float = 1.0):
    """启动全局性能监控

    Args:
        interval: 监控间隔（秒）
    """
    monitor = get_performance_monitor()
    monitor.start_monitoring(interval)


def stop_performance_monitoring():
    """停止全局性能监控"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


@contextmanager
def monitor_operation(operation_name: str):
    """操作监控上下文管理器

    Args:
        operation_name: 操作名称

    Yields:
        无
    """
    monitor = get_performance_monitor()
    with monitor.start_operation_timer(operation_name) as timer:
        yield timer


def get_performance_summary() -> Dict[str, Any]:
    """获取性能摘要

    Returns:
        性能摘要字典
    """
    monitor = get_performance_monitor()
    return monitor.get_performance_summary()
