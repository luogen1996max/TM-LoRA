import pandas as pd
import numpy as np

# 读取 CSV 文件

name = '龙珠'
qiuhe = 100

df = pd.read_csv(f'{name}1.csv')
df2 = pd.read_csv(f'{name}2.csv')

df_list = [i for i in df.iloc[:20001:,1]]
df_list2 = [i for i in df2.iloc[:20001:,1]]

s = pd.Series(df_list)
drop_points = s[s < s.shift(1)]
num = drop_points.keys()[0]

def csv_pro(df_list):
    a = df_list
    diffs = np.diff(a)
    result = diffs
    result_rounded = [round(x, 3) for x in result.tolist()]
    for i in range(98, len(result_rounded)+1, 100):
        result_rounded[i] = 0.5     ### 因为保存耗时，所以估计为0.5
    for i in range(num-1, len(result_rounded)+1, num):
        try:
            result_rounded[i] = 0.5     ### 断点检测
        except:
            pass
    result_rounded = [sum(result_rounded[i:i+qiuhe]) for i in range(0, len(result_rounded), qiuhe)]
    print(result_rounded)
    print(len(result_rounded))
    return result_rounded

result_rounded = csv_pro(df_list)
result_rounded2 = csv_pro(df_list2)


epoch_list = [i for i in range(1, len(result_rounded) + 1)]
import pandas as pd
record = dict()
record['epoch'] = epoch_list
record['our-Time'] = result_rounded
record['other-Time'] = result_rounded2
record = pd.DataFrame(record)
record_name = f"Time-{name}.csv"
record.to_csv(r'./%s' % (record_name), index=False)
#
#
print("=== 断点检测为： ===", num)
print("=== 此时的值为 ===", df.iloc[num-1, 1])
print("=== 下一个值为 ===", df.iloc[num, 1])