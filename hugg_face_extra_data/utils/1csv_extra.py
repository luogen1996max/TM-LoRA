import pandas as pd
import numpy as np

# 读取 CSV 文件

name = '瑞克'
qiuhe = 100

df = pd.read_csv(f'our_1.csv')
df2 = pd.read_csv(f'normal_1.csv')


df_list_lpips = [i for i in df.iloc[::,1]]
df_list_ssim = [i for i in df.iloc[::,2]]
###
df_list_lpips2 = [i for i in df2.iloc[::,1]]
df_list_ssim2 = [i for i in df2.iloc[::,2]]
###
result_rounded_lpips = [ round(i, 3) for i in df_list_lpips ]
result_rounded_ssim = [ round(i, 3) for i in df_list_ssim ]
###
result_rounded_lpips2 = [ round(i, 3) for i in df_list_lpips2 ]
result_rounded_ssim2 = [ round(i, 3) for i in df_list_ssim2 ]

epoch_list = [i for i in range(1, len(result_rounded_lpips) + 1)]
import pandas as pd
record = dict()
record['epoch'] = epoch_list
record['our-lpips'] = result_rounded_lpips
record['our-ssim'] = result_rounded_ssim
record['other-lpips'] = result_rounded_lpips2
record['other-ssim'] = result_rounded_ssim2
record = pd.DataFrame(record)
record_name = f"LPIPS-SSIM-{name}.csv"
record.to_csv(r'./%s' % (record_name), index=False)