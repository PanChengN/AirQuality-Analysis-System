"""
将四个城市的数据分别保存为单独的CSV文件

作者: niupancheng
创建日期：2025-06-12
最后修改日期：2025-06-12

此脚本用于：
1. 读取合并后的城市数据
2. 将每个城市的数据分别保存为单独的CSV文件

主要功能：
- 读取包含多个城市数据的CSV文件
- 识别文件中的不同城市
- 为每个城市创建单独的CSV文件
- 使用UTF-8编码保存文件，确保中文正确显示
"""

import pandas as pd

def separate_cities_data(input_file):
    """
    将合并的城市数据分离为单独的CSV文件
    
    参数:
    input_file: 输入的合并数据CSV文件路径
    """
    # 读取合并后的数据
    df = pd.read_csv(input_file)
    
    # 获取所有城市名称
    cities = df['city'].unique()
    
    # 为每个城市创建单独的CSV文件
    for city in cities:
        # 筛选当前城市的数据
        city_df = df[df['city'] == city]
        
        # 生成输出文件名
        output_file = f'{city}_数据.csv'
        
        # 保存为CSV文件
        city_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f'{city}的数据已保存到文件 {output_file}')

# 处理文件
separate_cities_data('所有城市数据.csv')
