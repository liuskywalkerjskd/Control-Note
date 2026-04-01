"""
Shared Chinese matplotlib configuration for figure generation.
Import this module to get Chinese font support:
    from zh_config import apply_zh, ZH_FONT
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

_FONT_PATH = os.path.expanduser('~/.fonts/NotoSansSC.ttf')
fm.fontManager.addfont(_FONT_PATH)
ZH_FONT = fm.FontProperties(fname=_FONT_PATH)
ZH_FONT_NAME = ZH_FONT.get_name()

def apply_zh():
    """Apply Chinese font globally to matplotlib rcParams."""
    plt.rcParams.update({
        'font.sans-serif': [ZH_FONT_NAME, 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'savefig.dpi': 200,
    })

# Translation dictionaries for common labels
LABELS = {
    'Time (s)': '时间 (s)',
    'Time (ms)': '时间 (ms)',
    'Angle (°)': '角度 (°)',
    'Position': '位置',
    'Position (m)': '位置 (m)',
    'Amplitude': '幅值',
    'Frequency (Hz)': '频率 (Hz)',
    'Frequency (rad/s)': '频率 (rad/s)',
    'Magnitude (dB)': '幅值 (dB)',
    'Speed (RPM)': '转速 (RPM)',
    'Current (A)': '电流 (A)',
    'Bias (°/s)': '偏置 (°/s)',
    'Control output': '控制输出',
    'Force (N)': '力 (N)',
    'Tilt (°)': '倾斜角 (°)',
}
