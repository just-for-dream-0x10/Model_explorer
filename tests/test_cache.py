"""
缓存系统单元测试
"""

import pytest
import time
from unittest.mock import patch
from utils.cache import CacheManager


class TestCacheManager:
    """缓存管理器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.cache = CacheManager(max_size=100, ttl=60)

    def test_basic_set_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        assert self.cache.get("nonexistent") is None
        assert self.cache.get("nonexistent", "default") == "default"

    def test_cache_expiration(self):
        """测试缓存过期"""
        # 设置短TTL的缓存
        short_cache = CacheManager(ttl=1)
        short_cache.set("expiring_key", "value")

        # 立即获取应该成功
        assert short_cache.get("expiring_key") == "value"

        # 等待过期
        time.sleep(1.1)
        assert short_cache.get("expiring_key") is None

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        # 创建小容量缓存
        small_cache = CacheManager(max_size=2)

        # 添加3个元素，应该淘汰第一个
        small_cache.set("key1", "value1")
        small_cache.set("key2", "value2")
        small_cache.set("key3", "value3")

        assert small_cache.get("key1") is None
        assert small_cache.get("key2") == "value2"
        assert small_cache.get("key3") == "value3"

    def test_cache_clear(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.clear()

        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        assert self.cache.size == 0

    def test_cache_delete(self):
        """测试删除特定键"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.delete("key1")

        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
        assert self.cache.size == 1

    def test_cache_stats(self):
        """测试缓存统计"""
        # 初始状态
        stats = self.cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # 添加数据
        self.cache.set("key1", "value1")

        # 命中
        self.cache.get("key1")

        # 未命中
        self.cache.get("nonexistent")

        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cached_decorator(self):
        """测试缓存装饰器"""
        call_count = 0

        @self.cache.cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # 第一次调用应该执行函数
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # 第二次调用应该从缓存获取
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # 没有增加

        # 不同参数应该重新执行
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading

        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                self.cache.set(key, value)
                retrieved = self.cache.get(key)
                assert retrieved == value

        # 创建多个线程同时访问缓存
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证缓存状态
        assert self.cache.size == 50  # 5个线程 × 10个操作

    def test_complex_data_types(self):
        """测试复杂数据类型的缓存"""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }

        self.cache.set("complex", complex_data)
        retrieved = self.cache.get("complex")

        assert retrieved == complex_data
        # 注意：set会被转换为list，因为JSON序列化限制
        assert isinstance(retrieved["set"], list)
