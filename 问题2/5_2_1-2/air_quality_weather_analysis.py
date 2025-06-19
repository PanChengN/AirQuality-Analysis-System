"""
城市A空气质量与气象关系建模分析
功能：分析城市A空气质量与气象因素的关系，建立预测模型，预测未来空气质量

作者：niupancheng
创建日期：2025-06-12
最后修改日期：2025-06-12

代码说明：
1. 数据预处理：加载历史数据和未来数据，进行必要的预处理
2. 相关性分析：分析气象因素与污染物之间的相关性，生成热力图
3. 模型训练：使用线性回归模型，考虑时间滞后效应
4. 模型评估：使用MSE、MAE、R²等指标评估模型性能
5. 未来预测：使用训练好的模型预测未来空气质量
6. 可视化：生成相关性热力图和预测结果图
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ==================== 配置参数 ====================
# 文件路径配置
DATA_FILES = {
    'historical': "../../数据预处理/城市A数据_1.csv",  # 历史数据文件路径
    'future': "../../数据预处理/城市A数据_2.csv"      # 未来数据文件路径
}

# 输出目录配置
OUTPUT_DIRS = {
    'figures': "figures",    # 图表输出目录
    'results': "results"     # 结果输出目录
}

# 数据列配置
CITY = 'A'                   # 城市标识
POLLUTANTS = ['SO2', 'NO2', 'PM10']  # 污染物指标
WEATHER = ['mmhg', 'tem', 'rh', 'ws']  # 气象指标

# 模型参数配置
MODEL_PARAMS = {
    'test_size': 0.2,        # 测试集比例
    'random_state': 42,      # 随机种子
    'n_lag': 3              # 时间滞后天数
}

# 可视化配置
JOURNAL_COLORS = {
    'primary': '#1f77b4',    # 主要数据线颜色
    'secondary': '#ff7f0e',  # 次要数据线颜色
    'grid': '#E0E0E0',       # 网格线颜色
    'background': '#FFFFFF', # 背景色
    'text': '#000000'        # 文字颜色
}

# ==================== 工具函数 ====================
def setup_environment():
    """创建必要的输出目录
    功能：确保输出目录存在，如果不存在则创建
    """
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

def setup_plotting_style():
    """设置全局绘图样式
    功能：配置matplotlib的绘图参数，使图表更美观
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',        # 设置字体
        'font.size': 12,               # 基础字体大小
        'axes.labelsize': 12,          # 轴标签字体大小
        'axes.titlesize': 14,          # 标题字体大小
        'xtick.labelsize': 10,         # x轴刻度标签字体大小
        'ytick.labelsize': 10,         # y轴刻度标签字体大小
        'legend.fontsize': 10,         # 图例字体大小
        'figure.titlesize': 14,        # 图形标题字体大小
        'axes.linewidth': 1.5,         # 轴线宽度
        'axes.grid': True,             # 显示网格
        'grid.alpha': 0.3,             # 网格透明度
        'lines.linewidth': 2,          # 线条宽度
        'lines.markersize': 6,         # 标记大小
        'savefig.dpi': 300,            # 保存图片分辨率
        'savefig.bbox': 'tight',       # 紧凑布局
        'savefig.pad_inches': 0.1,     # 边距
        'axes.spines.top': False,      # 隐藏上边框
        'axes.spines.right': False,    # 隐藏右边框
        'axes.axisbelow': True,        # 网格线在数据下方
        'axes.labelpad': 10,           # 轴标签间距
        'axes.titlepad': 15            # 标题间距
    })

def create_lag_features(df, target_col, feature_cols, n_lag=3):
    """创建滞后特征
    
    参数：
        df: DataFrame，必须包含target_col和feature_cols
        target_col: 目标变量列名
        feature_cols: 特征列名列表
        n_lag: 滞后天数
    
    返回：
        带有滞后特征的新DataFrame
    """
    df_new = df.copy()
    for lag in range(1, n_lag+1):
        for col in feature_cols + [target_col]:
            df_new[f'{col}_lag{lag}'] = df_new[col].shift(lag)
    return df_new.dropna().reset_index(drop=True)

def load_data():
    """加载并预处理数据
    
    功能：
        1. 检查数据文件是否存在
        2. 读取历史数据和未来数据
        3. 进行基本的数据预处理
    
    返回：
        tuple: (历史数据DataFrame, 未来数据DataFrame)
    """
    # 检查必要文件
    for file in DATA_FILES.values():
        if not os.path.exists(file):
            raise FileNotFoundError(f"错误：找不到文件 '{file}'。请确保该文件在正确的目录下。")
    
    # 读取历史数据
    df = pd.read_csv(DATA_FILES['historical'])
    df.columns = ['date'] + POLLUTANTS + WEATHER
    df['date'] = pd.to_datetime(df['date'])
    
    # 读取未来数据
    df_future = pd.read_csv(DATA_FILES['future'])
    df_future.columns = ['date'] + POLLUTANTS + WEATHER
    df_future['date'] = pd.to_datetime(df_future['date'])
    
    return df, df_future

def analyze_correlations(df):
    """分析相关性并生成热力图
    
    功能：
        1. 计算气象因素与污染物之间的相关性
        2. 生成相关性热力图
        3. 保存分析结果
    
    参数：
        df: 历史数据DataFrame
    """
    # 提取数据
    df_city = df[POLLUTANTS + WEATHER].copy()
    corr = df_city.corr().loc[WEATHER, POLLUTANTS]
    
    print("\n=== Correlation Analysis Results ===")
    print(f"\nCity {CITY} - Correlation Matrix:")
    print("Weather Variables vs Pollutants:")
    print(corr.round(3))
    print("-" * 50)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".2f", center=0,
                cbar_kws={'label': 'Correlation Coefficient', 'pad': 0.02},
                square=True, linewidths=0.5, ax=ax)
    plt.title(f"City {CITY} - Correlation between Weather and Pollutants", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIRS['figures']}/heatmap_corr_city_{CITY}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate_models(df):
    """训练和评估模型
    
    功能：
        1. 为每个污染物建立预测模型
        2. 使用时间滞后特征
        3. 评估模型性能
        4. 保存评估结果
    
    参数：
        df: 历史数据DataFrame
    
    返回：
        tuple: (模型字典, 标准化器字典, 评估结果DataFrame)
    """
    results = []
    model_dict = {}
    scaler_dict = {}
    
    # 提取数据
    df_city = df[POLLUTANTS + WEATHER].copy()
    
    for pol in POLLUTANTS:
        # 构造特征
        df_feat = create_lag_features(df_city, pol, WEATHER, MODEL_PARAMS['n_lag'])
        X = df_feat[[f'{w}_lag{l}' for l in range(1, MODEL_PARAMS['n_lag']+1) 
                    for w in WEATHER] + 
                   [f'{pol}_lag{l}' for l in range(1, MODEL_PARAMS['n_lag']+1)]].values
        y = df_feat[pol].values
        
        # 数据分割和标准化
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_PARAMS['test_size'], 
            random_state=MODEL_PARAMS['random_state']
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 保存结果
        results.append([CITY, pol, 'Linear', mse, mae, r2])
        model_dict[(CITY, pol, 'Linear')] = model
        scaler_dict[(CITY, pol)] = scaler
    
    results_df = pd.DataFrame(results, columns=['City', 'Pollutant', 'Model', 'MSE', 'MAE', 'R2'])
    results_df.to_csv(f"{OUTPUT_DIRS['results']}/model_evaluation_metrics.csv", index=False)
    
    return model_dict, scaler_dict, results_df

def predict_future(model_dict, scaler_dict, df_future):
    """预测未来数据
    
    功能：
        1. 使用训练好的模型预测未来空气质量
        2. 生成预测结果可视化
        3. 计算预测性能指标
        4. 保存预测结果
    
    参数：
        model_dict: 模型字典
        scaler_dict: 标准化器字典
        df_future: 未来数据DataFrame
    """
    future_results = []
    
    for pol in POLLUTANTS:
        # 构造特征
        df_future_feat = create_lag_features(df_future, pol, WEATHER, MODEL_PARAMS['n_lag'])
        X_future = df_future_feat[[f'{w}_lag{l}' for l in range(1, MODEL_PARAMS['n_lag']+1) 
                                 for w in WEATHER] + 
                                [f'{pol}_lag{l}' for l in range(1, MODEL_PARAMS['n_lag']+1)]].values
        y_true = df_future_feat[pol].values
        
        # 预测
        scaler = scaler_dict[(CITY, pol)]
        X_scaled = scaler.transform(X_future)
        model = model_dict[(CITY, pol, 'Linear')]
        y_pred = model.predict(X_scaled)
        
        # 绘制预测结果
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_future_feat['date'], y_true, label='Actual', 
                color=JOURNAL_COLORS['primary'], linewidth=2)
        ax.plot(df_future_feat['date'], y_pred, 
                label='Linear Regression Prediction',
                color=JOURNAL_COLORS['secondary'],
                linewidth=2, 
                linestyle='--')
        ax.set_title(f"Future Prediction - City {CITY} - {pol}", pad=20)
        plt.xticks(rotation=45)
        ax.set_ylabel(f"{pol} Concentration (μg/m³)", labelpad=10)
        ax.legend(frameon=True, fancybox=True, shadow=True, 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3, color=JOURNAL_COLORS['grid'])
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIRS['figures']}/predict_future_{CITY}_{pol}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        future_results.append([CITY, pol, 'Linear', mse, mae, r2])
    
    # 保存预测结果
    pd.DataFrame(future_results, columns=['City', 'Pollutant', 'Model', 'MSE', 'MAE', 'R2']).to_csv(
        f"{OUTPUT_DIRS['results']}/future_prediction_metrics.csv", index=False)

def main():
    """主函数
    
    功能：
        1. 初始化环境
        2. 加载数据
        3. 进行相关性分析
        4. 训练和评估模型
        5. 预测未来数据
        6. 输出分析结果
    """
    # 初始化环境
    setup_environment()
    setup_plotting_style()
    
    # 加载数据
    df, df_future = load_data()
    
    # 分析相关性
    analyze_correlations(df)
    
    # 训练和评估模型
    model_dict, scaler_dict, results_df = train_and_evaluate_models(df)
    
    # 打印模型评估结果
    print("\n=== Model Evaluation Metrics ===")
    print(f"\nCity {CITY}:")
    for pol in POLLUTANTS:
        print(f"\n  {pol}:")
        city_pol_results = results_df[(results_df['City'] == CITY) & 
                                    (results_df['Pollutant'] == pol)]
        for _, row in city_pol_results.iterrows():
            print(f"    {row['Model']} Model:")
            print(f"      MSE: {row['MSE']:.4f}")
            print(f"      MAE: {row['MAE']:.4f}")
            print(f"      R²:  {row['R2']:.4f}")
    
    # 预测未来数据
    predict_future(model_dict, scaler_dict, df_future)
    
    print("\n✅ Analysis completed. Results have been saved to the output directories.")

if __name__ == "__main__":
    main()
