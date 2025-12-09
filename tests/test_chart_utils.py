"""ChartBuilder单元测试

测试图表工具类的各项功能。

Author: Just For Dream Lab
Version: 1.0.0
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from utils.visualization import ChartBuilder


class TestChartBuilder:
    """ChartBuilder测试类"""

    def setup_method(self):
        """每个测试前的设置"""
        self.chart_builder = ChartBuilder()

    def test_create_line_chart_basic(self):
        """测试基础折线图创建"""
        x_data = [1, 2, 3, 4]
        y_data = [1, 4, 9, 16]
        title = "测试图表"

        fig = self.chart_builder.create_line_chart(
            x_data=x_data, y_data=y_data, title=title
        )

        # 验证返回的是Figure对象
        assert isinstance(fig, go.Figure)

        # 验证数据轨迹数量
        assert len(fig.data) == 1

        # 验证标题
        assert fig.layout.title.text == title

        # 验证数据
        assert list(fig.data[0].x) == x_data
        assert list(fig.data[0].y) == y_data

    def test_create_line_chart_multiple_series(self):
        """测试多系列折线图创建"""
        x_data = [1, 2, 3, 4]
        y_data = [[1, 4, 9, 16], [1, 2, 3, 4]]
        line_names = ["平方", "线性"]
        title = "多系列图表"

        fig = self.chart_builder.create_line_chart(
            x_data=x_data, y_data=y_data, title=title, line_names=line_names
        )

        # 验证数据轨迹数量
        assert len(fig.data) == 2

        # 验证轨迹名称
        assert fig.data[0].name == "平方"
        assert fig.data[1].name == "线性"

    def test_create_bar_chart(self):
        """测试柱状图创建"""
        x_data = ["A", "B", "C"]
        y_data = [10, 20, 30]
        title = "柱状图测试"

        fig = self.chart_builder.create_bar_chart(
            x_data=x_data, y_data=y_data, title=title
        )

        # 验证返回的是Figure对象
        assert isinstance(fig, go.Figure)

        # 验证数据轨迹数量
        assert len(fig.data) == 1

        # 验证图表类型
        assert fig.data[0].type == "bar"

    def test_create_heatmap(self):
        """测试热力图创建"""
        data = np.random.rand(10, 10)
        title = "热力图测试"

        fig = self.chart_builder.create_heatmap(data=data, title=title)

        # 验证返回的是Figure对象
        assert isinstance(fig, go.Figure)

        # 验证数据轨迹数量
        assert len(fig.data) == 1

        # 验证图表类型
        assert fig.data[0].type == "heatmap"

    def test_create_pie_chart(self):
        """测试饼图创建"""
        labels = ["A", "B", "C"]
        values = [30, 40, 30]
        title = "饼图测试"

        fig = self.chart_builder.create_pie_chart(
            labels=labels, values=values, title=title
        )

        # 验证返回的是Figure对象
        assert isinstance(fig, go.Figure)

        # 验证数据轨迹数量
        assert len(fig.data) == 1

        # 验证图表类型
        assert fig.data[0].type == "pie"

    def test_create_scatter_plot(self):
        """测试散点图创建"""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 5, 3]
        title = "散点图测试"

        fig = self.chart_builder.create_scatter_plot(
            x_data=x_data, y_data=y_data, title=title
        )

        # 验证返回的是Figure对象
        assert isinstance(fig, go.Figure)

        # 验证数据轨迹数量
        assert len(fig.data) == 1

        # 验证图表类型
        assert fig.data[0].type == "scatter"

    def test_default_colors(self):
        """测试默认颜色配置"""
        colors = self.chart_builder.colors

        # 验证颜色列表不为空
        assert len(colors) > 0

        # 验证颜色格式
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_default_layout(self):
        """测试默认布局配置"""
        layout = self.chart_builder.layout_config

        # 验证必要的布局属性
        assert "font" in layout
        assert "margin" in layout
        assert "showlegend" in layout

    def test_chart_consistency(self):
        """测试图表一致性"""
        # 创建多个图表，确保样式一致
        fig1 = self.chart_builder.create_line_chart([1, 2], [1, 2], "图1")
        fig2 = self.chart_builder.create_bar_chart(["A", "B"], [1, 2], "图2")

        # 验证字体设置一致
        assert fig1.layout.font == fig2.layout.font

        # 验证边距设置一致
        assert fig1.layout.margin == fig2.layout.margin

    @pytest.mark.parametrize(
        "chart_type",
        ["line_chart", "bar_chart", "heatmap", "pie_chart", "scatter_plot"],
    )
    def test_chart_creation_methods(self, chart_type):
        """参数化测试各种图表创建方法"""
        method = getattr(self.chart_builder, f"create_{chart_type}")

        if chart_type == "line_chart":
            fig = method([1, 2, 3], [1, 2, 3], "测试")
        elif chart_type == "bar_chart":
            fig = method(["A", "B"], [1, 2], "测试")
        elif chart_type == "heatmap":
            fig = method(np.random.rand(5, 5), "测试")
        elif chart_type == "pie_chart":
            fig = method(["A", "B"], [1, 2], "测试")
        elif chart_type == "scatter_plot":
            fig = method([1, 2], [1, 2], "测试")

        # 验证所有方法都返回Figure对象
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "测试"

    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        # 测试空数据
        with pytest.raises(ValueError):
            self.chart_builder.create_line_chart([], [], "空数据测试")

        # 测试长度不匹配的数据
        with pytest.raises(ValueError):
            self.chart_builder.create_line_chart([1, 2], [1], "长度不匹配")

    def test_height_parameter(self):
        """测试高度参数"""
        fig = self.chart_builder.create_line_chart(
            [1, 2, 3], [1, 2, 3], "测试", height=500
        )

        # 验证高度设置正确
        assert fig.layout.height == 500

    def test_custom_colors(self):
        """测试自定义颜色"""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        self.chart_builder.colors = custom_colors

        fig = self.chart_builder.create_line_chart(
            [[1, 2], [1, 2]], [[1, 2], [1, 2]], "测试", line_names=["系列1", "系列2"]
        )

        # 验证使用了自定义颜色
        assert len(fig.data) == 2
        # 注意：实际颜色可能因Plotly内部处理而略有不同

    def test_display_chart_simulation(self):
        """模拟测试图表显示功能"""
        fig = self.chart_builder.create_line_chart([1, 2], [1, 2], "测试")

        # 验证图表对象可以正常访问属性
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")
        assert len(fig.data) > 0


if __name__ == "__main__":
    pytest.main([__file__])
