"""
绘制城市A污染物浓度散点图矩阵
该脚本用于生成城市A的SO2、NO2和PM10三种污染物浓度的散点图矩阵，
展示污染物之间的相关性关系。

作者：niupancheng
创建日期：2024-03-21
最后修改日期：2024-03-21
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置 Matplotlib 全局字体与论文风格
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 11  # 适合期刊正文图注
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.edgecolor'] = 'black'

# 设置 Seaborn 样式
sns.set(style='white', palette='deep', color_codes=True)

# 读取数据
df = pd.read_csv('../../数据预处理/城市A_数据.csv')  # 含 SO2, NO2, PM10

# 画 Pairplot（散点图矩阵）
g = sns.pairplot(df[['SO2', 'NO2', 'PM10']],
                 kind='scatter',
                 plot_kws={'s': 25, 'edgecolor': 'k', 'linewidth': 0.5, 'alpha': 0.8},
                 diag_kws={'bins': 15, 'edgecolor': 'black', 'color': 'skyblue'})

# 标题与布局
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Scatter Plot Matrix of Pollutant Concentrations (City A)', fontsize=13)

# 去除多余白边并保存
plt.savefig('pollutants_scatter_cityA.pdf', dpi=300, bbox_inches='tight')
plt.close() 