"""
熵权法计算污染物权重
该脚本用于计算SO2、NO2、PM10三种污染物的权重系数

作者: niupancheng
创建日期: 2024-03-19
最后修改日期: 2024-03-19

功能说明:
1. 读取预处理后的城市污染物数据
2. 对数据进行极差标准化
3. 使用熵权法计算各污染物的权重
4. 输出权重结果并保存为CSV文件
"""

import pandas as pd
import numpy as np
import re

# 读取原始数据（包含 SO2, NO2, PM10 三列）
input_file = '../../数据预处理/城市D_数据.csv'
df = pd.read_csv(input_file)

# 从输入文件路径中提取城市标识
city_match = re.search(r'城市([A-D])_数据', input_file)
city_id = city_match.group(1) if city_match else 'unknown'

# 转换为 float 类型（自动跳过日期列）
df[['SO2', 'NO2', 'PM10']] = df[['SO2', 'NO2', 'PM10']].apply(pd.to_numeric, errors='coerce')

# 检查是否有缺失值
if df[['SO2', 'NO2', 'PM10']].isnull().values.any():
    print("⚠️ 存在无效值或空值，请检查数据源是否存在异常")

# 极差标准化
df_norm = (df[['SO2', 'NO2', 'PM10']] - df[['SO2', 'NO2', 'PM10']].min()) / \
          (df[['SO2', 'NO2', 'PM10']].max() - df[['SO2', 'NO2', 'PM10']].min())

# 步骤1：计算比例 p_ij
P = df_norm.div(df_norm.sum(axis=0), axis=1)  # 每列除以列和

# 步骤2：计算熵 e_j
n = len(P)
k = 1 / np.log(n)

# 避免 log(0)，加一个极小值
E = -k * (P * np.log(P + 1e-12)).sum(axis=0)

# 步骤3：计算差异系数 d_j
d = 1 - E

# 步骤4：计算权重 w_j
w = d / d.sum()

# 打印权重结果
weights_df = pd.DataFrame({
    '污染物': ['SO2', 'NO2', 'PM10'],
    '熵值 e_j': E.round(4).values,
    '差异系数 d_j': d.round(4).values,
    '权重 w_j': w.round(4).values
})

print(weights_df)

# 保存为 LaTeX 表格用 CSV
weights_df.to_csv(f'entropy_weights_city{city_id}.csv', index=False)