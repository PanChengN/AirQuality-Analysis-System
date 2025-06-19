"""
空气质量指数(AQI)排名方法
--------------------
本脚本实现了多种基于空气质量指数(AQI)数据的城市排名方法，包括：
1. 平均AQI排名
2. 分位数加权排名
3. 超标率加权排名
4. 稳定性加权排名
5. 综合排名（结合以上所有方法）

该脚本还包含使用matplotlib对排名结果进行可视化。

作者：niupancheng
创建日期：2024-03-19
最后修改：2024-03-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ---------- 1. 读取数据 ----------
df_raw = pd.read_csv('附件1.csv', skiprows=1)
df_raw.columns = ['date',
                  'A_SO2', 'A_NO2', 'A_PM10',
                  'B_SO2', 'B_NO2', 'B_PM10',
                  'C_SO2', 'C_NO2', 'C_PM10',
                  'D_SO2', 'D_NO2', 'D_PM10',
                  'mmhg', 'tem', 'rh', 'ws']
df_raw['date'] = pd.to_datetime(df_raw['date'])

# ---------- 2. 提取城市数据 ----------
cities = ['A', 'B', 'C', 'D']
data_dict = {}
for city in cities:
    df_city = df_raw[['date', f'{city}_SO2', f'{city}_NO2', f'{city}_PM10']].copy()
    df_city.columns = ['date', 'SO2', 'NO2', 'PM10']
    for col in ['SO2', 'NO2', 'PM10']:
        df_city[col] = pd.to_numeric(df_city[col], errors='coerce')
    data_dict[city] = df_city

# ---------- 3. 熵权法计算AQI ----------
def entropy_weight(df):
    df_std = (df - df.min()) / (df.max() - df.min())
    P = df_std / df_std.sum()
    P = P.replace(0, np.nan)
    n = df.shape[0]
    k = 1.0 / np.log(n)
    E = -k * (P * np.log(P)).sum()
    d = 1 - E
    w = d / d.sum()
    return w.values

# 计算每个城市的AQI
weights_dict = {}
AQI_results = {}
for city in cities:
    weights = entropy_weight(data_dict[city][['SO2', 'NO2', 'PM10']])
    weights_dict[city] = weights
    
    df_city = data_dict[city].copy()
    df_std = (df_city[['SO2', 'NO2', 'PM10']] - df_city[['SO2', 'NO2', 'PM10']].min()) / \
             (df_city[['SO2', 'NO2', 'PM10']].max() - df_city[['SO2', 'NO2', 'PM10']].min())
    df_city['AQI'] = df_std.mul(weights).sum(axis=1)
    AQI_results[city] = df_city

# ---------- 4. 合并所有城市的AQI数据 ----------
df_all = pd.DataFrame({'date': df_raw['date']})
for city in cities:
    df_all[f'City_{city}'] = AQI_results[city]['AQI']

# ---------- 5. 多种排名计算方法 ----------
def calculate_quantile_rank(df):
    """分位数加权平均法"""
    q75 = df.quantile(0.75)
    q50 = df.quantile(0.5)
    q25 = df.quantile(0.25)
    weighted_score = 0.4 * q75 + 0.3 * q50 + 0.3 * q25
    return weighted_score

def calculate_exceedance_rank(df, threshold=0.5):
    """超标率加权法"""
    exceedance_rate = (df > threshold).mean()
    mean_aqi = df.mean()
    # 综合评分：超标率权重0.6，平均AQI权重0.4
    weighted_score = 0.6 * exceedance_rate + 0.4 * mean_aqi
    return weighted_score

def calculate_stability_rank(df):
    """波动性加权法"""
    mean_aqi = df.mean()
    std_aqi = df.std()
    # 使用变异系数（标准差/均值）的倒数作为稳定性指标
    stability = 1 / (std_aqi / mean_aqi)
    # 综合评分：稳定性权重0.4，平均AQI权重0.6
    weighted_score = 0.4 * stability + 0.6 * mean_aqi
    return weighted_score

# ---------- 6. 计算并输出各种排名结果 ----------
# 准备数据
df_aqi = df_all.drop(columns='date')

# 计算各种排名
quantile_rank = calculate_quantile_rank(df_aqi)
exceedance_rank = calculate_exceedance_rank(df_aqi)
stability_rank = calculate_stability_rank(df_aqi)
mean_rank = df_aqi.mean()

# 创建结果DataFrame
results = pd.DataFrame({
    'Mean_AQI': mean_rank,
    'Quantile_Weighted': quantile_rank,
    'Exceedance_Weighted': exceedance_rank,
    'Stability_Weighted': stability_rank
})

# 对每种方法进行排名
rankings = results.rank(ascending=False)
rankings.columns = [col + '_Rank' for col in rankings.columns]

# 合并结果
final_results = pd.concat([results, rankings], axis=1)
print("\nAQI Ranking Results:")
print(final_results)

# ---------- 8. 计算综合排名 ----------
def calculate_comprehensive_rank(df):
    """计算综合排名
    权重分配：
    - 分位数加权: 0.35 (考虑整体分布)
    - 超标率加权: 0.30 (考虑高污染情况)
    - 稳定性加权: 0.20 (考虑稳定性)
    - 平均AQI: 0.15 (考虑整体水平)
    """
    # 标准化各个方法的得分
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    # 计算各个方法的标准化得分
    quantile_norm = normalize(calculate_quantile_rank(df))
    exceedance_norm = normalize(calculate_exceedance_rank(df))
    stability_norm = normalize(calculate_stability_rank(df))
    mean_norm = normalize(df.mean())
    
    # 计算综合得分
    comprehensive_score = (
        0.35 * quantile_norm +
        0.30 * exceedance_norm +
        0.20 * stability_norm +
        0.15 * mean_norm
    )
    
    return comprehensive_score

# 计算综合排名
comprehensive_rank = calculate_comprehensive_rank(df_aqi)
results['Comprehensive'] = comprehensive_rank

# ---------- 10. 可视化排名结果 ----------
# 设置期刊风格的图表样式
plt.style.use('seaborn-v0_8-whitegrid')
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
    'legend.frameon': True,
    'legend.fontsize': 9,
    'legend.edgecolor': 'black',
    'legend.framealpha': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 600
})

# 设置颜色方案（使用期刊风格的低饱和度配色）
colors = ['#4C72B0',  # 柔和的蓝色
          '#55A868',  # 柔和的绿色
          '#C44E52',  # 柔和的红色
          '#CCB974']  # 柔和的黄色（用于Stability Weighted）

# 图1：所有方法的对比
plt.figure(figsize=(8, 6))
x = np.arange(len(cities))
width = 0.2  # 恢复宽度以适应4个柱子

# 绘制柱状图（去掉Comprehensive）
bars1 = plt.bar(x - 1.5*width, results['Mean_AQI'], width, label='Mean AQI', color=colors[0])
bars2 = plt.bar(x - 0.5*width, results['Quantile_Weighted'], width, label='Quantile Weighted', color=colors[1])
bars3 = plt.bar(x + 0.5*width, results['Exceedance_Weighted'], width, label='Exceedance Weighted', color=colors[2])
bars4 = plt.bar(x + 1.5*width, results['Stability_Weighted'], width, label='Stability Weighted', color=colors[3])

# 设置样式
plt.xlabel('Cities', fontsize=11)
plt.ylabel('Score', fontsize=11)
plt.title('AQI Rankings by Different Methods', pad=15, fontsize=12)
plt.xticks(x, [f'City {city}' for city in cities])
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# 调整布局
plt.tight_layout()

# 保存第一张图
plt.savefig('AQI_ranking_comparison.pdf', 
            dpi=600, 
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()

# 图2：综合排名
plt.figure(figsize=(5, 3))  # 调整图形尺寸
y_pos = np.arange(len(cities))

# 绘制水平条形图
bars = plt.barh(y_pos, results['Comprehensive'], color='#8172B3', height=0.6)  # 调整条形图高度
plt.yticks(y_pos, [f'City {city}' for city in cities])
plt.xlabel('Comprehensive Score', fontsize=11)
plt.title('Comprehensive AQI Ranking', pad=15, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

# 添加数值标签，优化位置和样式
for i, v in enumerate(results['Comprehensive']):
    # 统一将标签放在条形图右侧
    plt.text(v + 0.02, i, f'{v:.4f}', 
             va='center', 
             ha='left',
             fontsize=9,  # 稍微减小字体
             color='black')  # 统一使用黑色

# 调整x轴范围，确保有足够空间显示标签
plt.xlim(0, max(results['Comprehensive']) * 1.25)  # 增加右侧空间

# 移除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存第二张图
plt.savefig('AQI_comprehensive_ranking.pdf', 
            dpi=600, 
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()

# ---------- 11. 打印排名顺序 ----------
def print_ranking_order(series, method_name):
    """Print ranking order from worst to best"""
    sorted_rank = series.sort_values(ascending=False)
    print(f"\n{method_name} Ranking (from worst to best):")
    for i, (city, score) in enumerate(sorted_rank.items(), 1):
        print(f"{i}. {city}: {score:.4f}")

# Print ranking orders
print_ranking_order(results['Mean_AQI'], "Mean AQI")
print_ranking_order(results['Quantile_Weighted'], "Quantile Weighted")
print_ranking_order(results['Exceedance_Weighted'], "Exceedance Weighted")
print_ranking_order(results['Stability_Weighted'], "Stability Weighted")
print_ranking_order(results['Comprehensive'], "Comprehensive") 
