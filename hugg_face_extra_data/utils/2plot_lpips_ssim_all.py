import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
name = '龙珠'
name1 = 'DragonBall-SS-Captions'
df = pd.read_csv(f'./LPIPS-SSIM-{name}.csv')

print("列名:", df.columns.tolist())  # 调试：查看列名

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(5, 5))
plt.figure(figsize=(8, 5))

colors = {
    # 第1对：红 vs 绿（经典对比色）
    'our1': '#FF0000',  # 纯红
    'other1': '#00AA00',  # 鲜绿

    # 第2对：橙 vs 蓝（互补色）
    'our2': '#FF8C00',  # 深橙
    'other2': '#0055FF',  # 亮蓝

    # # 第3对：紫 vs 黄绿（高反差）
    # 'our3': '#8B00FF',  # 紫罗兰
    # 'other3': '#55DD33',  # 黄绿
    #
    # # 第4对：深红 vs 青（强对比）
    # 'our4': '#CC0000',  # 深红
    # 'other4': '#00CCCC',  # 青
}


# 绘制两条曲线（使用原始数据，不移动平均）
plt.plot(df['epoch'], df['our-lpips'], '-', linewidth=2.5, label='our-lpips', color=colors['our1'])
plt.plot(df['epoch'], df['other-lpips'], '-.', alpha=0.6, linewidth=2.5, label='lpips', color=colors['other1'])

plt.plot(df['epoch'], df['our-ssim'], '-', linewidth=2.5, label='our-ssim', color=colors['our2'])
plt.plot(df['epoch'], df['other-ssim'], '-.', alpha=0.6, linewidth=2.5, label='other-ssim', color=colors['other2'])

# plt.plot(df['epoch'], df['our-Time3'], '-', linewidth=2.5, label='我们训练时间1', color=colors['our3'])
# plt.plot(df['epoch'], df['other-Time3'], '-.', alpha=0.3, linewidth=2.5, label='其他训练时间1', color=colors['other3'])
#
# plt.plot(df['epoch'], df['our-Time4'], '-', linewidth=2.5, label='我们训练时间1', color=colors['our4'])
# plt.plot(df['epoch'], df['other-Time4'], '-.', alpha=0.3, linewidth=2.5, label='其他训练时间1', color=colors['other4'])
# 设置横坐标刻度为整数
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 假设 epoch 是从 0 到 20 的整数
plt.xticks(range(0, int(df['epoch'].max()) + 1, 1))  # 每2个显示一个

plt.title(f'{name1}数据集', fontsize=14, fontweight='bold')
plt.xlabel('20个模型的lpips-ssim', fontsize=12)
plt.ylabel('时间(S)', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# # 添加注释（根据你的图片）
# plt.annotate('(x, y) = (18.30, 0.412)',
#              xy=(18.30, 0.412),
#              xytext=(15, 0.45),
#              arrowprops=dict(arrowstyle='->', color='gray'),
#              fontsize=10)

plt.tight_layout()
plt.savefig(f'line_chart_smooth{name}.png', dpi=300)
plt.show()