import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 绘制 FID-τ 折线图 ====================
# 数据
tau_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
fid_17 = [0.565, 0.560, 0.550, 0.540, 0.530, 0.520]
fid_17_5 = [0.550, 0.540, 0.530, 0.525, 0.520, 0.510]
fid_18 = [0.530, 0.520, 0.510, 0.500, 0.490, 0.480]
fid_18_5 = [0.520, 0.510, 0.500, 0.490, 0.480, 0.470]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制折线
ax.plot(tau_values, fid_17, 'o-', linewidth=2, markersize=8, label='17.0')
ax.plot(tau_values, fid_17_5, 's-', linewidth=2, markersize=8, label='17.5')
ax.plot(tau_values, fid_18, '^-', linewidth=2, markersize=8, label='18.0')
ax.plot(tau_values, fid_18_5, 'd-', linewidth=2, markersize=8, label='18.5')

# 设置标签和标题
ax.set_xlabel('τ', fontsize=14, fontweight='bold')
ax.set_ylabel('FID', fontsize=14, fontweight='bold')
ax.set_title('FID vs τ在不同参数下的变化', fontsize=16, fontweight='bold')

# 设置图例
ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.6)

# 设置坐标轴范围
ax.set_xlim(0.95, 1.55)
ax.set_ylim(0.45, 0.58)

# 添加数值标签
for i, (x, y) in enumerate(zip(tau_values, fid_17)):
    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fid_tau_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 2. 绘制 LPIPS vs SSIM 散点图 ====================
# 数据
lpips_data = [
    0.533131485, 0.557377659, 0.593665367, 0.577802742, 0.56948045,
    0.58480379, 0.612343568, 0.585616206, 0.596599039, 0.597612619,
    0.588328885, 0.597497526, 0.601302439, 0.604389942, 0.598574769,
    0.586798519, 0.588483612, 0.593472205, 0.598666728, 0.587719604
]

ssim_data = [
    0.427782388, 0.415697837, 0.396338726, 0.408543003, 0.43501878,
    0.413599631, 0.39717759, 0.424750442, 0.418611802, 0.417753635,
    0.416748409, 0.411830655, 0.411106197, 0.409027733, 0.407363927,
    0.418363034, 0.415298433, 0.406211338, 0.411882498, 0.412780573
]

fig2, ax2 = plt.subplots(figsize=(10, 8))

# 绘制散点
scatter = ax2.scatter(lpips_data, ssim_data, c='steelblue', s=80,
                      edgecolors='white', linewidth=1.5, alpha=0.8)

# 添加拟合趋势线
z = np.polyfit(lpips_data, ssim_data, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(lpips_data), max(lpips_data), 100)
ax2.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'趋势线 (斜率={z[0]:.3f})')

# 设置标签和标题
ax2.set_xlabel('LPIPS', fontsize=14, fontweight='bold')
ax2.set_ylabel('SSIM', fontsize=14, fontweight='bold')
ax2.set_title('LPIPS vs SSIM 散点图', fontsize=16, fontweight='bold')

# 添加图例
ax2.legend(loc='upper right', fontsize=12)

# 添加网格
ax2.grid(True, linestyle='--', alpha=0.6)

# 设置坐标轴范围
ax2.set_xlim(0.50, 0.65)
ax2.set_ylim(0.38, 0.45)

# 添加均值线
ax2.axhline(y=np.mean(ssim_data), color='green', linestyle=':', alpha=0.7, label=f'SSIM均值: {np.mean(ssim_data):.3f}')
ax2.axvline(x=np.mean(lpips_data), color='orange', linestyle=':', alpha=0.7, label=f'LPIPS均值: {np.mean(lpips_data):.3f}')
ax2.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig('lpips_ssim_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 3. 绘制组合图（类似论文风格）====================
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：FID-τ 折线图
ax3.plot(tau_values, fid_17, 'o-', linewidth=2, markersize=6, label='17.0', color='#1f77b4')
ax3.plot(tau_values, fid_17_5, 's-', linewidth=2, markersize=6, label='17.5', color='#ff7f0e')
ax3.plot(tau_values, fid_18, '^-', linewidth=2, markersize=6, label='18.0', color='#2ca02c')
ax3.plot(tau_values, fid_18_5, 'd-', linewidth=2, markersize=6, label='18.5', color='#d62728')
ax3.set_xlabel('τ', fontsize=12, fontweight='bold')
ax3.set_ylabel('FID', fontsize=12, fontweight='bold')
ax3.set_title('(a) FID vs τ', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.5)

# 右图：LPIPS vs SSIM 散点图
scatter = ax4.scatter(lpips_data, ssim_data, c='steelblue', s=60,
                       edgecolors='black', linewidth=0.5, alpha=0.7)
ax4.set_xlabel('LPIPS', fontsize=12, fontweight='bold')
ax4.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax4.set_title('(b) LPIPS vs SSIM', fontsize=14, fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.5)

# 添加趋势线
z = np.polyfit(lpips_data, ssim_data, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(lpips_data), max(lpips_data), 100)
ax4.plot(x_trend, p(x_trend), 'r-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 所有图表已生成完成！")