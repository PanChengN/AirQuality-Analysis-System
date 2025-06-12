"""
作者：niupancheng
创建日期：2024-03-21
最后修改日期：2024-03-21

所有城市空气质量数据预处理脚本

此脚本用于处理所有城市的空气质量监测数据，主要功能包括：
1. 从原始Excel文件中提取所有城市的相关数据
2. 处理数据格式，包括列名重命名和数据结构优化
3. 将处理后的数据保存为CSV格式文件

数据包含以下指标：
- SO2: 二氧化硫浓度
- NO2: 二氧化氮浓度
- PM10: 可吸入颗粒物浓度
"""

import pandas as pd

def process_all_cities_data(file_path, output_file):
    """
    处理所有城市的数据并保存为CSV
    
    参数:
    file_path: 输入Excel文件路径
    output_file: 输出CSV文件名
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 定义城市及其对应的列
    city_columns = {
        '城市A': [1, 2, 3],  # SO2, NO2, PM10的列索引
        '城市B': [4, 5, 6],
        '城市C': [7, 8, 9],
        '城市D': [10, 11, 12]
    }
    
    # 创建结果DataFrame
    result_dfs = []
    
    for city, cols in city_columns.items():
        # 提取当前城市的相关列
        columns_to_keep = [df.columns[0]]  # 日期列
        columns_to_keep.extend([df.columns[i] for i in cols])
        
        # 选择列
        df_city = df[columns_to_keep]
        
        # 删除第一行（原来的列名行）
        df_city = df_city.iloc[1:]
        
        # 设置新的列名
        df_city.columns = ['data', 'SO2', 'NO2', 'PM10']
        
        # 添加城市列
        df_city['city'] = city
        
        # 将日期列转换为datetime格式
        df_city['data'] = pd.to_datetime(df_city['data'])
        
        result_dfs.append(df_city)
    
    # 合并所有城市的数据
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    # 按城市和日期排序
    final_df = final_df.sort_values(['city', 'data'])
    
    # 保存到CSV文件
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'所有城市的数据已保存到文件 {output_file}')
    
    return final_df

# 处理文件
df = process_all_cities_data(
    file_path='../附件1_原格式.xls',
    output_file='所有城市数据.csv'
) 