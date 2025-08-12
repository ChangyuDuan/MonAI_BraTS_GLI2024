import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
import tempfile
import msvcrt
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
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
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
    执行字体配置
    """
    global _font_configured
    
    # Windows系统中文字体列表（按优先级排序）
    font_candidates = [
        'Microsoft YaHei',      # 微软雅黑
        'Microsoft YaHei UI',   # 微软雅黑UI
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'KaiTi',                # 楷体
        'FangSong',             # 仿宋
        'Microsoft JhengHei',   # 微软正黑体
        'DengXian',             # 等线
        'YouYuan',              # 幼圆
        'LiSu',                 # 隶书
        'STXihei',              # 华文细黑
        'STKaiti',              # 华文楷体
        'STSong',               # 华文宋体
        'STFangsong',           # 华文仿宋
        'NSimSun'               # 新宋体
    ]
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break
    
    # 如果没有找到预定义的字体，尝试查找包含中文字符的字体
    if selected_font is None:
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name
            # 检查字体名称是否包含中文字体相关关键词
            chinese_keywords = [
                '中文', '黑体', '宋体', '楷体', '仿宋', '雅黑', '等线', '幼圆', '隶书',
                'Microsoft', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'YaHei',
                'JhengHei', 'DengXian', 'YouYuan', 'LiSu', 'STXihei', 'STKaiti',
                'STSong', 'STFangsong', 'NSimSun', 'CJK', 'Han', 'Hei', 'Song'
            ]
            if any(keyword in font_name for keyword in chinese_keywords):
                chinese_fonts.append(font_name)
        
        if chinese_fonts:
            selected_font = chinese_fonts[0]
    
    # 配置matplotlib
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        print(f"✓ 已配置中文字体: {selected_font}")
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
        
        print("未找到中文字体，建议安装中文字体包")
        _font_configured = True  # 标记已尝试配置，避免重复警告
        return None