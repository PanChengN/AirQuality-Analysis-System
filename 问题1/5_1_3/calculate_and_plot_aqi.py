"""
计算并绘制四个城市的空气质量指数(AQI)时间序列图

该脚本用于：
1. 读取四个城市(A-D)的污染物数据
2. 计算每个城市的AQI指数
3. 生成包含四个子图的时间序列图，展示各城市AQI的变化趋势

作者：niupancheng
创建日期：2024-03-21
最后修改日期：2024-03-21
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- 更新图像全局样式设置 ----------
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 600
})

# 定义权重
weights = [0.4699, 0.2713, 0.2588]

# 标准化处理
def standardize(x):
    return (x - x.mean()) / x.std()

# 计算单个城市的 AQI
def calculate_aqi(df):
    # 对每个污染物进行标准化
    df['SO2_std'] = standardize(df['SO2'])
    df['NO2_std'] = standardize(df['NO2'])
    df['PM10_std'] = standardize(df['PM10'])
    
    # 计算 AQI
    df['AQI'] = (df['SO2_std'] * weights[0] + 
                 df['NO2_std'] * weights[1] + 
                 df['PM10_std'] * weights[2])
    return df

# 读取所有城市的数据并计算 AQI
cities = ['A', 'B', 'C', 'D']
dfs = {}

for city in cities:
    file_path = f'../../数据预处理/城市{city}_数据.csv'
    df = pd.read_csv(file_path)
    dfs[city] = calculate_aqi(df)

# 设置颜色方案（使用期刊风格的低饱和度配色）
colors = ['#4C72B0',  # 柔和的蓝色
          '#55A868',  # 柔和的绿色
          '#C44E52',  # 柔和的红色
          '#8172B3']  # 柔和的紫色

# 创建一个大图，包含4个子图
fig, axes = plt.subplots(4, 1, figsize=(10, 12))
fig.suptitle('Daily AQI of Cities A-D (2009/6 - 7)', y=0.98, fontsize=14)

# 绘制每个城市的子图
for i, (city, ax) in enumerate(zip(cities, axes)):
    ax.plot(pd.to_datetime(dfs[city]['data']), 
            dfs[city]['AQI'], 
            marker='o',
            markersize=3,
            color=colors[i])
    ax.set_title(f'City {city}', pad=10, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 只在最后一个子图显示x轴标签
    if i == len(cities) - 1:
        ax.set_xlabel('Date', fontsize=11)
    
    # 所有子图都显示y轴标签
    ax.set_ylabel('AQI', fontsize=11)
    
    # 调整x轴日期格式
    ax.tick_params(axis='x', rotation=45)

# 调整子图之间的间距和整体布局
plt.tight_layout()
# 为顶部标题留出更多空间
plt.subplots_adjust(top=0.92)

# 保存图片
plt.savefig('aqi_fourcities_subplots.pdf', 
            dpi=600, 
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()