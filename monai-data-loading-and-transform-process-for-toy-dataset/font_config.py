# -*- coding: utf-8 -*-
"""
中文字体配置模块
解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings
from pathlib import Path

def configure_chinese_font():
    """
    配置matplotlib中文字体支持
    
    自动检测系统可用的中文字体并配置matplotlib
    解决中文乱码、不显示或显示为小方块的问题
    """
    
    # 获取系统类型
    system = platform.system()
    
    # 定义不同系统的中文字体列表（按优先级排序）
    font_candidates = {
        'Windows': [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong'              # 仿宋
        ],
        'Darwin': [  # macOS
            'PingFang SC',          # 苹方
            'Heiti SC',             # 黑体-简
            'STHeiti',              # 华文黑体
            'Arial Unicode MS'      # Arial Unicode MS
        ],
        'Linux': [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Source Han Sans CN',   # 思源黑体
            'DejaVu Sans'           # DejaVu Sans
        ]
    }
    
    # 获取当前系统的字体候选列表
    candidates = font_candidates.get(system, font_candidates['Windows'])
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font_name in candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break
    
    # 如果没有找到预定义的字体，尝试查找包含中文字符的字体
    if selected_font is None:
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name
            # 检查字体名称是否包含中文相关关键词
            chinese_keywords = ['中文', '黑体', '宋体', '楷体', '仿宋', '雅黑', 'CJK', 'Han', 'Hei', 'Song']
            if any(keyword in font_name for keyword in chinese_keywords):
                chinese_fonts.append(font_name)
        
        if chinese_fonts:
            selected_font = chinese_fonts[0]
    
    # 配置matplotlib
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        print(f"✓ 已配置中文字体: {selected_font}")
        return selected_font
    else:
        # 如果仍然没有找到合适的字体，使用备用方案
        warnings.warn(
            "未找到合适的中文字体，可能会出现中文显示问题。\n"
            "建议安装中文字体或使用英文标签。",
            UserWarning
        )
        
        # 设置基本配置
        plt.rcParams['axes.unicode_minus'] = False
        
        print("⚠ 未找到中文字体，建议安装中文字体包")
        return None

def get_font_suggestions():
    """
    获取字体安装建议
    
    Returns:
        str: 字体安装建议文本
    """
    system = platform.system()
    
    suggestions = {
        'Windows': """
        Windows系统字体建议：
        1. 系统通常自带微软雅黑、黑体等中文字体
        2. 如果缺少字体，可以从控制面板 > 字体中安装
        3. 推荐字体：Microsoft YaHei, SimHei, SimSun
        """,
        
        'Darwin': """
        macOS系统字体建议：
        1. 系统通常自带苹方、黑体-简等中文字体
        2. 可以从字体册应用中管理字体
        3. 推荐字体：PingFang SC, Heiti SC, STHeiti
        """,
        
        'Linux': """
        Linux系统字体建议：
        1. 安装文泉驿字体：sudo apt-get install fonts-wqy-microhei
        2. 安装思源字体：sudo apt-get install fonts-noto-cjk
        3. 更新字体缓存：fc-cache -fv
        4. 推荐字体：WenQuanYi Micro Hei, Noto Sans CJK SC
        """
    }
    
    return suggestions.get(system, suggestions['Linux'])

def test_chinese_display():
    """
    测试中文字体显示效果
    
    创建一个简单的图表来验证中文字体是否正常显示
    """
    import numpy as np
    
    # 配置中文字体
    font_name = configure_chinese_font()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.arange(5)
    y = np.random.rand(5)
    
    # 绘制图表
    bars = ax.bar(x, y, color=['red', 'green', 'blue', 'orange', 'purple'])
    
    # 设置中文标签
    ax.set_title('中文字体测试图表', fontsize=16, fontweight='bold')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['第一类', '第二类', '第三类', '第四类', '第五类'])
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存测试图片
    test_path = Path('font_test.png')
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    
    print(f"\n字体测试完成！")
    print(f"测试图片已保存到: {test_path.absolute()}")
    
    if font_name:
        print(f"当前使用字体: {font_name}")
        print("如果图片中的中文正常显示，说明字体配置成功！")
    else:
        print("未找到合适的中文字体，图片中的中文可能显示为方块。")
        print(get_font_suggestions())
    
    plt.show()
    
    return test_path

if __name__ == "__main__":
    print("=" * 60)
    print("matplotlib中文字体配置工具")
    print("=" * 60)
    
    # 测试字体配置
    test_chinese_display()
    
    print("\n" + "=" * 60)
    print("配置完成！在其他模块中调用 configure_chinese_font() 即可启用中文支持")
    print("=" * 60)