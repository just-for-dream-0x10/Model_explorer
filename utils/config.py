"""
系统配置和字体设置
"""

import locale
import sys
import platform
import matplotlib.pyplot as plt


def detect_chinese_support():
    """检测系统是否支持中文显示"""
    try:
        # 检测系统语言环境
        system_language = locale.getdefaultlocale()[0]
        if system_language and "zh" in system_language.lower():
            return True

        # 检测系统编码
        if sys.getdefaultencoding().lower().startswith("utf"):
            return True

        # 尝试显示中文字符
        test_str = "测试"
        test_str.encode(sys.getdefaultencoding())
        return True
    except:
        return False


def configure_matplotlib_font():
    """配置matplotlib字体以支持中文"""
    chinese_supported = detect_chinese_support()

    if chinese_supported:
        try:
            # 尝试设置中文字体
            if platform.system() == "Darwin":  # macOS
                plt.rcParams["font.sans-serif"] = [
                    "Arial Unicode MS",
                    "PingFang SC",
                    "SimHei",
                    "Microsoft YaHei",
                ]
            elif platform.system() == "Windows":
                plt.rcParams["font.sans-serif"] = [
                    "SimHei",
                    "Microsoft YaHei",
                    "Arial Unicode MS",
                ]
            else:  # Linux
                plt.rcParams["font.sans-serif"] = [
                    "DejaVu Sans",
                    "SimHei",
                    "Arial Unicode MS",
                ]

            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "测试", fontsize=12)
            plt.close(fig)
            return True

        except:
            # 如果中文字体设置失败，使用英文
            plt.rcParams["font.sans-serif"] = [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
            ]
            return False
    else:
        # 系统不支持中文，使用英文字体
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
        return False


# 执行字体配置
CHINESE_SUPPORTED = configure_matplotlib_font()
