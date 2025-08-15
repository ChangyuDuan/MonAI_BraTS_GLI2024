import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings
import os
import tempfile
from pathlib import Path


# 全局标志，防止重复配置
_font_configured = False
# 使用临时文件作为锁文件
_FONT_LOCK_FILE = os.path.join(tempfile.gettempdir(), 'matplotlib_chinese_font.lock')

def configure_chinese_font():
    """
    配置matplotlib中文字体支持
    自动检测系统可用的中文字体并配置matplotlib
    解决中文乱码、不显示或显示为小方块的问题
    """
    global _font_configured
    
    # 检查是否已经配置过
    if _font_configured:
        return None
    
    # 检查锁文件是否存在（跨进程检查）
    if os.path.exists(_FONT_LOCK_FILE):
        _font_configured = True
        return None
    
    # 尝试创建锁文件
    try:
        with open(_FONT_LOCK_FILE, 'w') as lock_file:
            # 使用Windows文件锁
            import msvcrt
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except (OSError, ImportError):
                # 如果无法获取锁，说明其他进程正在配置
                _font_configured = True
                return None
            
            # 获取锁成功，执行配置
            lock_file.write('font_configured')
            lock_file.flush()
            
            return _do_font_configuration()
    except Exception:
        # 如果创建锁文件失败，直接执行配置（降级处理）
        return _do_font_configuration()

def _do_font_configuration():
    """
    实际执行字体配置的函数
    """
    global _font_configured
    
    # 清除matplotlib字体缓存
    try:
        import matplotlib
        matplotlib.font_manager._rebuild()
    except Exception:
        pass
    
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
        ]
    }
    
    # 获取当前系统的字体候选列表
    candidates = font_candidates.get(system, font_candidates['Windows'])
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"系统可用字体总数: {len(available_fonts)}")
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font_name in candidates:
        if font_name in available_fonts:
            selected_font = font_name
            print(f"找到候选字体: {font_name}")
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
            print(f"找到备选中文字体: {selected_font}")
    
    # 配置matplotlib
    if selected_font:
        # 强制设置字体，清除原有配置
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 验证字体是否真正可用
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print(f"✓ 已配置并验证中文字体: {selected_font}")
        except Exception as e:
            print(f"字体验证失败: {e}")
        
        _font_configured = True  # 标记已配置
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
        _font_configured = True  # 标记已尝试配置，避免重复警告
        return None

def force_configure_chinese_font():
    """
    强制配置中文字体（每次绘图前调用）
    解决seaborn等库重置matplotlib字体配置的问题
    
    Returns:
        str: 配置的字体名称，如果配置失败返回None
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
    
    # 强制配置matplotlib字体
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return selected_font
    else:
        # 设置基本配置
        plt.rcParams['axes.unicode_minus'] = False
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