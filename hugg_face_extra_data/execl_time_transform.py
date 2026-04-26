import pandas as pd

# 读取 CSV 文件



df = pd.read_csv('a.csv')

df_list = [i for i in df.iloc[::,1]]
# 创建 Series
s = pd.Series(df_list)
# 方法1：找下降点（当前值 < 前一个值）
drop_points = s[s < s.shift(1)]


num = drop_points.keys()[0]

print("=== 断点检测为： ===", num)
print("=== 此时的值为 ===", df.iloc[num-1, 1])
print("=== 下一个值为 ===", df.iloc[num, 1])
# if len(drop_points) > 0:
#     for idx, val in drop_points.items():
#         print(f"断点位置: 索引 {idx} → {idx-1} → {idx}")
#         print(f"  前一个值 ({idx-1}): {s[idx-1]:.6f}")
#         print(f"  当前值 ({idx}):   {val:.6f}")
#         print(f"  下降幅度: {s[idx-1] - val:.6f}")
# else:
#     print("未发现下降断点")

# print(df.iloc[num-1::num,1])
# 每隔1224行取 B 列的值（假设 B 列是第二列，索引为1）
total = df.iloc[num-1:20001:num, 1].sum()
print(f"总和: {total}")