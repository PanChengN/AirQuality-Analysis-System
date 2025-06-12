"""
城市A空气质量数据预处理脚本

此脚本用于处理城市A的空气质量监测数据，主要功能包括：
1. 从两个原始Excel文件中提取城市A的相关数据
2. 处理数据格式，包括列名重命名和数据结构优化
3. 将处理后的数据分别保存为两个CSV格式文件
4. 合并两个CSV文件并保存为新的合并文件

数据包含以下指标：
- SO2: 二氧化硫浓度
- NO2: 二氧化氮浓度
- PM10: 可吸入颗粒物浓度
- mmhg: 气压
- tem: 温度
- rh: 相对湿度
- ws: 风速

作者：niupancheng
创建日期：2024-03-21
最后修改日期：2024-03-21
"""

import pandas as pd

def process_city_a_data(file_path, output_file, column_indices):
    """
    处理城市A的数据并保存为CSV
    
    参数:
    file_path: 输入Excel文件路径
    output_file: 输出CSV文件名
    column_indices: 包含气象参数列索引的元组
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 提取城市A的相关列
    columns_to_keep = [
        'Unnamed: 0',  # 日期列
        '城市A', 'Unnamed: 2', 'Unnamed: 3',  # 城市A的污染物数据
        '城市A的气象参数', f'Unnamed: {column_indices[0]}', 
        f'Unnamed: {column_indices[1]}', f'Unnamed: {column_indices[2]}'  # 城市A的气象参数
    ]
    
    # 选择列
    df_city_a = df[columns_to_keep]
    
    # 删除第一行（原来的列名行）
    df_city_a = df_city_a.iloc[1:]
    
    # 设置新的列名
    df_city_a.columns = ['data', 'SO2', 'NO2', 'PM10', 'mmhg', 'tem', 'rh', 'ws']
    
    # 将日期列转换为datetime格式并排序
    df_city_a['data'] = pd.to_datetime(df_city_a['data'])
    df_city_a = df_city_a.sort_values('data')
    
    # 保存到新的CSV文件
    df_city_a.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'城市A的数据已保存到文件 {output_file}')
    
    return df_city_a

def merge_city_a_data(df1, df2, output_file):
    """
    合并两个城市A的数据文件
    
    参数:
    df1: 第一个数据文件的数据框
    df2: 第二个数据文件的数据框
    output_file: 输出合并后的CSV文件名
    """
    # 合并两个数据框
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 按日期排序并删除重复行
    merged_df = merged_df.sort_values('data')
    merged_df = merged_df.drop_duplicates(subset=['data'])
    
    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'合并后的城市A数据已保存到文件 {output_file}')

# 处理第一个文件
df1 = process_city_a_data(
    file_path='../附件1_原格式.xls',
    output_file='城市A数据_1.csv',
    column_indices=(14, 15, 16)
)

# 处理第二个文件
df2 = process_city_a_data(
    file_path='../附件2_原格式.xls',
    output_file='城市A数据_2.csv',
    column_indices=(5, 6, 7)
)

# 合并两个文件
merge_city_a_data(df1, df2, '城市A数据_合并.csv')