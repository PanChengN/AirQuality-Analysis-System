"""
该脚本从CSV文件中读取污染物数据，绘制原始和标准化后的箱线图，并将图表保存为PDF文件。

作者: niupancheng
创建日期：2025-06-12
最后修改日期：2025-06-12

功能说明:
1. 读取城市A的污染物数据（SO2、NO2、PM10）
2. 绘制原始数据的箱线图
3. 对数据进行MinMax标准化处理
4. 绘制标准化后的箱线图
5. 将图表保存为PDF格式
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv('../../数据预处理/城市A数据_1.csv', parse_dates=['data'])

# 原始污染物数据
pollutants = data[['SO2', 'NO2', 'PM10']]

# 原始数据箱线图
plt.figure(figsize=(8, 5))
pollutants.boxplot()
plt.title('Pollutants in City A (Raw Data)')
plt.ylabel('Concentration')
plt.savefig('pollutants_raw_boxplot.pdf', dpi=300)
plt.close()

# 标准化处理
scaler = MinMaxScaler()
pollutants_scaled = pd.DataFrame(scaler.fit_transform(pollutants), columns=pollutants.columns)

# 标准化后箱线图
plt.figure(figsize=(8, 5))
pollutants_scaled.boxplot()
plt.title('Pollutants in City A (Scaled Data)')
plt.ylabel('Normalized Concentration')
plt.savefig('pollutants_scaled_boxplot.pdf', dpi=300)
plt.close()
