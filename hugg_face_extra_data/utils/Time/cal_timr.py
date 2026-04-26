import pandas as pd

# 读取 CSV 文件



df = pd.read_csv('xxx.csv')

df_list = [i for i in df.iloc[:20001:,1]]
# 创建 Series
s = pd.Series(df_list)
# 方法1：找下降点（当前值 < 前一个值）
drop_points = s[s < s.shift(1)]
num = drop_points.keys()[0]



import numpy as np
a = df_list
diffs = np.diff(a)
result = diffs

# result = diffs[diffs >= 0]
# result = [a[i:i+num] for i in range(0, len(a), num)]
# result = np.concatenate([np.diff(i) for i in result]).tolist()

result_rounded = [round(x, 3) for x in result.tolist()]
print(result[198])
for i in range(98, len(result_rounded)+1, 100):
    result_rounded[i] = 0.5     ### 因为保存耗时，所以估计为0.5

for i in range(num-1, len(result_rounded)+1, num):
    result_rounded[i] = 0.5     ### 断点检测

result_rounded = [sum(result_rounded[i:i+1000]) for i in range(0, len(result_rounded), 1000)]

print(result_rounded)
print(len(result_rounded))

epoch_list = [i for i in range(1, len(result_rounded)+1)]
import pandas as pd
record = dict()
record['epoch'] = epoch_list
record['Time'] = result_rounded
record = pd.DataFrame(record)
record_name = "time1.csv"
record.to_csv(r'./%s' % (record_name), index=False)
#
#
print("=== 断点检测为： ===", num)
print("=== 此时的值为 ===", df.iloc[num-1, 1])
print("=== 下一个值为 ===", df.iloc[num, 1])