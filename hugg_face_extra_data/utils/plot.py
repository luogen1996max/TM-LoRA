import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_excel('./zzz火影.xlsx',
                   engine='openpyxl',
                   header=None,
                   skiprows=1,
                   names=['class', 'our-lpips', 'our-ssim', 'other-lpips', 'other-ssim'])

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 6))

# 定义颜色（可根据喜好调整）
colors = {
    'our': '#E74C3C',      # 红色
    'SD': '#3498DB',       # 蓝色
    'TOME': '#2ECC71',     # 绿色
    'Fourier': '#F39C12'   # 橙色
}

# 移动平均平滑函数
def moving_average(data, window=20):
    return data.rolling(window=window, center=True).mean()

# 绘制原始折线（半透明，使用对应颜色）
plt.plot(df['class'], df['our-lpips'], 'o-', alpha=0.3, linewidth=1, markersize=2, color=colors['our'])
plt.plot(df['class'], df['our-ssim'], 's-', alpha=0.3, linewidth=1, markersize=2, color=colors['SD'])
plt.plot(df['class'], df['other-lpips'], '^-', alpha=0.3, linewidth=1, markersize=2, color=colors['TOME'])
plt.plot(df['class'], df['other-ssim'], 'd-', alpha=0.3, linewidth=1, markersize=2, color=colors['Fourier'])

# 绘制平滑曲线（使用相同颜色，但不透明）
plt.plot(df['class'], moving_average(df['our-lpips'], 50), '-', linewidth=2.5, label='我们LPIPS', color=colors['our'])
plt.plot(df['class'], moving_average(df['our-ssim'], 50), '-', linewidth=2.5, label='我们SSIM', color=colors['SD'])
plt.plot(df['class'], moving_average(df['other-lpips'], 50), '-', linewidth=2.5, label='传统LPIPS', color=colors['TOME'])
plt.plot(df['class'], moving_average(df['other-ssim'], 50), '-', linewidth=2.5, label='传统SSIM', color=colors['Fourier'])

plt.xlabel('20个模型的', fontsize=12)
plt.ylabel('评估分值', fontsize=12)
# plt.title('各类Token合并时间对比', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('line_chart_smooth.png', dpi=300)
plt.show()