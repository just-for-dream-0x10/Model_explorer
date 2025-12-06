"""缓存管理模块

提供高效的缓存机制，优化重复计算的性能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import functools
import hashlib
import json
import time
from typing import Any, Dict, Optional, Callable, Union
import threading

from .exceptions import CacheError


class CacheManager:
    """缓存管理器
    
    提供线程安全的缓存功能，支持TTL和LRU策略。
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认TTL（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._access_order = []  # LRU访问顺序
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或过期则返回None
            
        Raises:
            CacheError: 当缓存操作失败时
        """
        try:
            with self._lock:
                if key not in self._cache:
                    return None
                
                entry = self._cache[key]
                
                # 检查是否过期
                if self._is_expired(entry):
                    self._remove_entry(key)
                    return None
                
                # 更新访问时间
                entry['accessed_at'] = time.time()
                self._update_access_order(key)
                
                return entry['value']
        except Exception as e:
            raise CacheError("get", key, str(e))
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），None表示使用默认值
            
        Raises:
            CacheError: 当缓存操作失败时
        """
        try:
            with self._lock:
                # 如果缓存已满，移除最旧的条目
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_lru()
                
                current_time = time.time()
                ttl = ttl if ttl is not None else self.default_ttl
                
                self._cache[key] = {
                    'value': value,
                    'created_at': current_time,
                    'accessed_at': current_time,
                    'ttl': ttl,
                    'expires_at': current_time + ttl
                }
                
                self._update_access_order(key)
        except Exception as e:
            raise CacheError("set", key, str(e))
    
    def delete(self, key: str) -> bool:
        """删除缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
            
        Raises:
            CacheError: 当缓存操作失败时
        """
        try:
            with self._lock:
                if key in self._cache:
                    self._remove_entry(key)
                    return True
                return False
        except Exception as e:
            raise CacheError("delete", key, str(e))
    
    def clear(self) -> None:
        """清空所有缓存
        
        Raises:
            CacheError: 当缓存操作失败时
        """
        try:
            with self._lock:
                self._cache.clear()
                self._access_order.clear()
        except Exception as e:
            raise CacheError("clear", "all", str(e))
    
    def cleanup_expired(self) -> int:
        """清理过期的缓存条目
        
        Returns:
            清理的条目数量
            
        Raises:
            CacheError: 当缓存操作失败时
        """
        try:
            with self._lock:
                expired_keys = [
                    key for key, entry in self._cache.items()
                    if self._is_expired(entry)
                ]
                
                for key in expired_keys:
                    self._remove_entry(key)
                
                return len(expired_keys)
        except Exception as e:
            raise CacheError("cleanup_expired", "all", str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        with self._lock:
            total_entries = len(self._cache)
            expired_count = sum(
                1 for entry in self._cache.values()
                if self._is_expired(entry)
            )
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_count,
                'valid_entries': total_entries - expired_count,
                'max_size': self.max_size,
                'usage_ratio': total_entries / self.max_size
            }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查缓存条目是否过期
        
        Args:
            entry: 缓存条目
            
        Returns:
            是否过期
        """
        return time.time() > entry['expires_at']
    
    def _remove_entry(self, key: str) -> None:
        """移除缓存条目
        
        Args:
            key: 缓存键
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _update_access_order(self, key: str) -> None:
        """更新LRU访问顺序
        
        Args:
            key: 缓存键
        """
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """移除最久未使用的缓存条目"""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)


# 全局缓存管理器实例
_global_cache = CacheManager()


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器
    
    Returns:
        缓存管理器实例
    """
    return _global_cache


def cache_key(*args, **kwargs) -> str:
    """生成缓存键
    
    Args:
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        缓存键字符串
    """
    # 将参数序列化为字符串
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    # 使用MD5哈希生成固定长度的键
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    cache_manager: Optional[CacheManager] = None
):
    """缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        key_func: 自定义键生成函数
        cache_manager: 缓存管理器，None表示使用全局实例
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache_key(func.__name__, *args, **kwargs)
            
            # 获取缓存管理器
            manager = cache_manager or get_cache_manager()
            
            # 尝试从缓存获取
            cached_result = manager.get(key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            manager.set(key, result, ttl)
            
            return result
        
        # 添加缓存管理方法
        wrapper.cache_clear = lambda: manager.clear()
        wrapper.cache_delete = lambda k: manager.delete(k)
        wrapper.cache_stats = lambda: manager.get_stats()
        
        return wrapper
    
    return decorator


def cached_method(ttl: Optional[int] = None):
    """方法缓存装饰器
    
    专门用于类方法的缓存，自动处理self参数。
    
    Args:
        ttl: 缓存生存时间（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成包含类信息的缓存键
            class_name = self.__class__.__name__
            method_name = func.__name__
            key = cache_key(class_name, method_name, *args, **kwargs)
            
            manager = get_cache_manager()
            
            # 尝试从缓存获取
            cached_result = manager.get(key)
            if cached_result is not None:
                return cached_result
            
            # 执行方法并缓存结果
            result = func(self, *args, **kwargs)
            manager.set(key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


# 预定义的键生成函数
def network_analysis_key(input_shape: tuple, layers: list) -> str:
    """网络分析缓存键生成函数
    
    Args:
        input_shape: 输入形状
        layers: 层配置列表
        
    Returns:
        缓存键
    """
    return cache_key("network_analysis", input_shape, layers)


def param_calculation_key(layer_type: str, params: dict) -> str:
    """参数计算缓存键生成函数
    
    Args:
        layer_type: 层类型
        params: 层参数
        
    Returns:
        缓存键
    """
    return cache_key("param_calculation", layer_type, params)


# 定期清理任务
def start_cleanup_task(interval: int = 300) -> None:
    """启动定期清理任务
    
    Args:
        interval: 清理间隔（秒）
    """
    import threading
    
    def cleanup_task():
        while True:
            try:
                manager = get_cache_manager()
                cleaned = manager.cleanup_expired()
                if cleaned > 0:
                    print(f"清理了 {cleaned} 个过期缓存条目")
            except Exception as e:
                print(f"缓存清理失败: {e}")
            
            time.sleep(interval)
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()


# 启动清理任务
start_cleanup_task()