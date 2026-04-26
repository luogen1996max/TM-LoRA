import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
name = 'all'
df = pd.read_csv(f'./Time-{name}.csv')

print("列名:", df.columns.tolist())  # 调试：查看列名

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 6))

colors = {
    # 第1对：红 vs 绿（经典对比色）
    'our1': '#FF0000',  # 纯红
    'other1': '#00AA00',  # 鲜绿

    # 第2对：橙 vs 蓝（互补色）
    'our2': '#FF8C00',  # 深橙
    'other2': '#0055FF',  # 亮蓝

    # 第3对：紫 vs 黄绿（高反差）
    'our3': '#8B00FF',  # 紫罗兰
    'other3': '#55DD33',  # 黄绿

    # 第4对：深红 vs 青（强对比）
    'our4': '#CC0000',  # 深红
    'other4': '#00CCCC',  # 青
}



# 绘制两条曲线（使用原始数据，不移动平均）
plt.plot(df['epoch'], df['our-Time1'], '-', linewidth=1.5, label='我们训练时间1', color=colors['our1'])
plt.plot(df['epoch'], df['other-Time1'], '-.', alpha=0.5, linewidth=2.5, label='其他训练时间1', color=colors['other1'])

plt.plot(df['epoch'], df['our-Time2'], '-', linewidth=1.5, label='我们训练时间2', color=colors['our2'])
plt.plot(df['epoch'], df['other-Time2'], '-.', alpha=0.5, linewidth=2.5, label='其他训练时间2', color=colors['other2'])

plt.plot(df['epoch'], df['our-Time3'], '-', linewidth=1.5, label='我们训练时间3', color=colors['our3'])
plt.plot(df['epoch'], df['other-Time3'], '-.', alpha=0.5, linewidth=2.5, label='其他训练时间3', color=colors['other3'])

plt.plot(df['epoch'], df['our-Time4'], '-', linewidth=1.5, label='我们训练时间4', color=colors['our4'])
plt.plot(df['epoch'], df['other-Time4'], '-.', alpha=0.5, linewidth=2.5, label='其他训练时间4', color=colors['other4'])

plt.xlabel('总2000个epochs-每间隔100epoch求总时长', fontsize=12)
plt.ylabel('时间(S)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('line_chart_smooth.png', dpi=300)
plt.show()