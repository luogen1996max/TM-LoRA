import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
name = '火影'
df = pd.read_csv(f'./Time-{name}.csv')

print("列名:", df.columns.tolist())  # 调试：查看列名

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 6))

colors = {
    'our': '#E74C3C',      # 红色
    'other': '#3498DB',    # 蓝色
}

# 绘制两条曲线（使用原始数据，不移动平均）
plt.plot(df['epoch'], df['our-Time'], '-', linewidth=2.5, label='我们训练时间', color=colors['our'])
plt.plot(df['epoch'], df['other-Time'], '-', linewidth=2.5, label='其他训练时间', color=colors['other'])

plt.xlabel('总2000个epochs-每间隔100epoch求总时长', fontsize=12)
plt.ylabel('时间(S)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('line_chart_smooth.png', dpi=300)
plt.show()