"""
文件名: grey_correlation_analysis.py
作者: niupancheng
创建日期：2025-06-12
最后修改日期：2025-06-12

代码说明:
本程序实现了基于灰色关联分析的空气质量预测模型。主要功能包括：
1. 使用灰色关联度矩阵对气象因素进行加权
2. 对数据进行标准化预处理
3. 构建神经网络模型进行多污染物预测
4. 进行模型训练和评估
5. 可视化预测结果和误差分布

主要步骤：
1. 数据读取和预处理
2. 灰色关联度权重计算
3. 特征工程和标准化
4. 神经网络模型构建和训练
5. 预测结果评估和可视化

依赖库：
- pandas: 数据处理
- numpy: 数值计算
- sklearn: 数据标准化和评估指标
- torch: 深度学习框架
- matplotlib: 数据可视化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

# 设置颜色方案（使用期刊风格的低饱和度配色）
colors = ['#4C72B0',  # 柔和的蓝色
          '#55A868',  # 柔和的绿色
          '#C44E52',  # 柔和的红色
          '#8172B3']  # 柔和的紫色

# --- 1. 读取数据 ---
train_df = pd.read_csv('../../数据预处理/城市A数据_1.csv', parse_dates=['data'])
test_df = pd.read_csv('../../数据预处理/城市A数据_2.csv', parse_dates=['data'])

# --- 2. 定义灰色关联度矩阵 ---
gra_matrix = {
    'SO2': [0.671, 0.640, 0.639, 0.587],
    'NO2': [0.615, 0.681, 0.632, 0.643],
    'PM10': [0.630, 0.682, 0.660, 0.588]
}
features = ['tem', 'rh', 'ws', 'mmhg']

# --- 3. 计算归一化权重 ---
gra_weights = {}
for pollutant, vals in gra_matrix.items():
    vals = np.array(vals)
    weights = vals / vals.sum()
    gra_weights[pollutant] = weights
print("灰色关联度归一化权重:\n", gra_weights)

# --- 4. 数据预处理 ---
# 标准化气象变量
scaler_x = StandardScaler()
train_x_raw = train_df[features].values
test_x_raw = test_df[features].values

# 用训练集数据fit scaler
train_x_scaled = scaler_x.fit_transform(train_x_raw)
test_x_scaled = scaler_x.transform(test_x_raw)

# 标准化污染物（目标）
target_cols = ['SO2', 'NO2', 'PM10']
scaler_y = StandardScaler()
train_y_raw = train_df[target_cols].values
test_y_raw = test_df[target_cols].values
train_y_scaled = scaler_y.fit_transform(train_y_raw)
test_y_scaled = scaler_y.transform(test_y_raw)

# --- 5. 加权气象变量 ---
# 对每个污染物分别加权气象输入，构造多输出模型的输入特征
# 这里示范方式为加权气象数据乘以对应权重后合并三组输入特征（示范）
# 你也可以为每个污染物单独训练模型或其他方式

def weighted_features(x_scaled, weights):
    return x_scaled * weights

# 为简化示例，假设训练一个多输出模型，输入为所有加权特征拼接
train_x_list = []
test_x_list = []
for pollutant in target_cols:
    w = gra_weights[pollutant]
    train_x_list.append(weighted_features(train_x_scaled, w))
    test_x_list.append(weighted_features(test_x_scaled, w))

train_x_final = np.hstack(train_x_list)
test_x_final = np.hstack(test_x_list)

# --- 6. 构建神经网络模型 ---
class AirQualityNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 7. 转换为Tensor和DataLoader ---
train_dataset = TensorDataset(torch.tensor(train_x_final, dtype=torch.float32),
                              torch.tensor(train_y_scaled, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_x_final, dtype=torch.float32),
                             torch.tensor(test_y_scaled, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --- 8. 训练模型 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_mse = float('inf')
best_model_state = None
best_predictions = None

# 存储每次训练的MSE结果
all_mse_results = {pollutant: [] for pollutant in target_cols}
# 存储每次训练的预测结果
all_predictions = {pollutant: [] for pollutant in target_cols}

print("\n=== Training Progress ===")
for run in range(10):
    print(f"\nTraining Run {run + 1}/10")
    model = AirQualityNN(input_dim=train_x_final.shape[1], output_dim=len(target_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        if (epoch+1) % 40 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # 评估当前模型
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(test_x_final, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test).cpu().numpy()
    
    # 反标准化预测值
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = test_y_raw
    
    # 存储每次训练的预测结果
    for i, pollutant in enumerate(target_cols):
        all_predictions[pollutant].append(y_pred[:, i])
        current_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        all_mse_results[pollutant].append(current_mse)
        print(f"{pollutant} MSE: {current_mse:.6f}")
    
    # 计算总体MSE
    current_mse = mean_squared_error(y_true, y_pred)
    print(f"Overall MSE: {current_mse:.6f}")
    
    # 如果当前模型更好，保存它
    if current_mse < best_mse:
        best_mse = current_mse
        best_model_state = model.state_dict().copy()
        best_predictions = y_pred.copy()
        print(f"New best model found! MSE: {best_mse:.6f}")

# 计算并显示统计结果
print("\n=== Statistical Results ===")
for pollutant in target_cols:
    mse_values = all_mse_results[pollutant]
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    print(f"{pollutant}:")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Std MSE: {std_mse:.6f}")
    print(f"Min MSE: {min(mse_values):.6f}")
    print(f"Max MSE: {max(mse_values):.6f}")
    print()

print(f"\nBest overall model MSE: {best_mse:.6f}")

# 使用最佳模型的预测结果
y_pred = best_predictions
y_true = test_y_raw

# 创建预测结果的可视化
plt.figure(figsize=(15, 10))

# 为每个污染物创建子图
for i, pollutant in enumerate(target_cols):
    # 创建实际值vs预测值的对比图
    plt.subplot(3, 2, i*2 + 1)
    
    # 计算所有训练运行的预测均值和标准差
    predictions_array = np.array(all_predictions[pollutant])
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)
    
    # 绘制实际值
    plt.plot(test_df['data'], y_true[:, i], 'b-', label='Actual', alpha=0.7, color=colors[0])
    
    # 绘制预测均值和标准差范围
    plt.plot(test_df['data'], mean_predictions, 'r--', label='Mean Predicted', alpha=0.7, color=colors[1])
    plt.fill_between(test_df['data'], 
                     mean_predictions - std_predictions,
                     mean_predictions + std_predictions,
                     color=colors[1], alpha=0.2, label='Predicted ±1σ')
    
    plt.title(f'{pollutant} Actual vs Predicted Values', pad=15, fontsize=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Concentration', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 创建预测误差分布图
    plt.subplot(3, 2, i*2 + 2)
    
    # 计算所有训练运行的误差均值和标准差
    errors_array = np.array(all_predictions[pollutant]) - y_true[:, i]
    mean_errors = np.mean(errors_array, axis=0)
    std_errors = np.std(errors_array, axis=0)
    
    # 绘制误差均值和标准差范围
    plt.plot(test_df['data'], mean_errors, 'g-', label='Mean Error', alpha=0.7, color=colors[2])
    plt.fill_between(test_df['data'], 
                     mean_errors - std_errors,
                     mean_errors + std_errors,
                     color=colors[2], alpha=0.2, label='Error ±1σ')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(f'{pollutant} Prediction Error Distribution', pad=15, fontsize=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Error', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_visualization.pdf', 
            dpi=600, 
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()

# 打印每天的预测结果
print("\n=== Prediction Results Details ===")
for i, pollutant in enumerate(target_cols):
    print(f"\n{pollutant} Prediction Results:")
    print("Date\t\tActual\t\tMean Predicted\tStd Dev\t\tMean Absolute Error")
    print("-" * 70)
    mean_pred = np.mean(all_predictions[pollutant], axis=0)
    std_pred = np.std(all_predictions[pollutant], axis=0)
    for j in range(len(test_df)):
        actual = y_true[j, i]
        predicted = mean_pred[j]
        std = std_pred[j]
        # 计算所有预测值与实际值之间的平均绝对误差
        abs_errors = np.abs(np.array(all_predictions[pollutant])[:, j] - actual)
        mean_abs_error = np.mean(abs_errors)
        print(f"{test_df['data'].iloc[j].strftime('%Y-%m-%d')}\t{actual:.4f}\t\t{predicted:.4f}\t\t{std:.4f}\t\t{mean_abs_error:.4f}")

# 打印总体评估指标
print("\n=== Overall Evaluation Metrics ===")
for i, pollutant in enumerate(target_cols):
    mean_pred = np.mean(all_predictions[pollutant], axis=0)
    mse = mean_squared_error(y_true[:, i], mean_pred)
    mae = mean_absolute_error(y_true[:, i], mean_pred)
    print(f"{pollutant} Performance: MSE={mse:.6f}, MAE={mae:.6f}")
