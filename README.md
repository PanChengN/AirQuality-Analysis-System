# 城市空气质量分析与预测系统

## 项目背景

大气是地球生态系统的重要组成部分，直接影响人类健康和社会发展。研究表明，人类可以五周不进食或五天不饮水，但若超过五分钟无法呼吸空气，生命便会受到严重威胁。然而，随着工业化和城市化进程加快，大气污染问题日益突出，主要污染物如 SO₂（二氧化硫）、NO₂（氮氧化物）和PM10（可吸入颗粒物）的浓度升高，不仅危害人体健康，还可能加剧全球气候变化。

目前，许多城市已建立空气质量监测体系，但如何从长期监测数据中挖掘污染规律、预测未来趋势，并制定科学的防控策略，仍是环境治理的关键问题。此外，气象条件（如温度、湿度、风速、气压等）对污染物扩散具有重要影响，因此，研究空气质量与气象参数的关系，对污染预警和治理具有重要意义。

## 问题描述

本项目基于A、B、C、D四个城市的空气质量监测数据进行分析和建模。数据包括：
- 附件1：四个城市从2009年6月1日至2009年7月25日期间的空气质量监测数据，包括 SO₂、NO₂、PM10的浓度以及城市A的气象参数（温度、湿度、风速、气压等）
- 附件2：城市A从2009年7月26日至2009年7月30日期间的空气质量监测数据

主要解决以下两个问题：
1. 建立空气质量评价模型，并根据附件1提供的数据对四个城市的空气质量进行排序
2. 分析附件1中城市A的空气质量与气象参数之间的关系，建立模型并利用附件2中的数据进行检验

## 项目结构

```
.
├── 数据预处理/                # 数据预处理相关脚本
│   ├── all_cities_pollutant_weather_preprocessing.py    # 所有城市数据预处理
│   ├── cityA_pollutant_weather_preprocessing_2.py       # 城市A数据预处理
│   └── separate_cities_data.py                          # 城市数据分离
│
├── 问题1/                    # 问题1相关分析
│   ├── 5_1_1/               # 污染物分析
│   │   ├── pollutants_scatter_cityA.py    # 城市A污染物散点图
│   │   └── plot_pollutant_boxplots.py     # 污染物箱线图
│   ├── 5_1_2/               # 熵权法分析
│   │   └── entropy_weight_calculation.py
│   └── 5_1_3/               # 城市AQI分析
│       ├── calculate_and_plot_aqi.py
│       └── AQI_ranking_methods.py
│
└── 问题2/                    # 问题2相关分析
    ├── 5_2_1-2/             # 城市A空气质量与气象关系建模
    │   └── air_quality_weather_analysis.py
    └── 5_2_3/               # 灰色关联分析
        └── grey_correlation_analysis.py
```

## 功能特点

### 1. 数据预处理
- **多城市数据预处理**
  - 数据格式标准化
  - 缺失值处理与填充
  - 异常值检测与处理
  - 数据分离与合并
  - 时间序列数据对齐

### 2. 空气质量分析
- **污染物浓度分析**
  - 污染物浓度时间序列分析
  - 污染物浓度分布特征分析
  - 污染物浓度相关性分析
- **可视化分析**
  - 污染物浓度箱线图
  - 污染物散点图矩阵
  - 污染物浓度热力图
  - 污染物浓度趋势图

### 3. AQI计算与评估
- **基于熵权法的AQI计算**
  - 数据标准化处理
  - 熵值计算
  - 权重确定
  - AQI综合评分
- **多城市AQI对比分析**
  - 城市间AQI对比
  - 污染物贡献度分析
  - 空气质量排名方法实现

### 4. 预测模型
- **线性回归模型**
  - 多元线性回归
  - 模型参数估计
  - 模型诊断与评估
- **灰色关联分析**
  - 灰色关联度计算
  - 影响因素排序
  - 关联度可视化
- **非线性加权建模**
  - 神经网络模型
  - 模型训练与优化
  - 预测结果评估

## 环境要求

- Python 3.7+
- 依赖包：
  - pandas (>= 1.3.0) - 数据处理和分析
  - numpy (>= 1.20.0) - 数值计算
  - matplotlib (>= 3.4.0) - 数据可视化
  - seaborn (>= 0.11.0) - 统计可视化
  - scikit-learn (>= 0.24.0) - 机器学习算法
  - torch (>= 1.9.0) - 深度学习框架
  - scipy (>= 1.7.0) - 科学计算
  - statsmodels (>= 0.13.0) - 统计模型

## 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理
```bash
cd 数据预处理
# 处理所有城市数据
python all_cities_pollutant_weather_preprocessing.py
# 处理城市A数据
python cityA_pollutant_weather_preprocessing_2.py
# 分离城市数据
python separate_cities_data.py
```

### 2. 问题1分析
```bash
cd 问题1
# 污染物分析
cd 5_1_1
python pollutants_scatter_cityA.py
python plot_pollutant_boxplots.py

# 熵权法分析
cd ../5_1_2
python entropy_weight_calculation.py

# AQI分析
cd ../5_1_3
python calculate_and_plot_aqi.py
python AQI_ranking_methods.py
```

### 3. 问题2分析
```bash
cd 问题2
# 空气质量与气象关系建模
cd 5_2_1-2
python air_quality_weather_analysis.py

# 灰色关联分析
cd ../5_2_3
python grey_correlation_analysis.py
```

## 输出说明

### 1. 数据文件
- 预处理后的数据保存在 `数据预处理/processed_data/` 目录
- 分析结果保存在各脚本所在目录的 `results/` 文件夹
- 所有数据文件采用CSV格式，使用UTF-8编码

### 2. 可视化结果
- 部分图表保存在各脚本所在目录的 `figures/` 文件夹
- 支持多种格式：PDF（用于论文）、PNG（用于网页）、SVG（用于矢量图）
- 图表分辨率：300dpi，适合出版要求

### 3. 模型评估结果
- 模型性能指标保存在 `results/model_metrics/` 目录
- 包含：R²、RMSE、MAE等评估指标
- 预测结果保存在 `results/predictions/` 目录

## 注意事项

1. **数据准备**
   - 确保数据文件位于正确的目录
   - 检查数据格式是否符合要求
   - 注意数据编码（UTF-8）

2. **环境配置**
   - 建议使用虚拟环境
   - 确保所有依赖包版本正确
   - GPU加速需要CUDA支持

3. **运行顺序**
   - 严格按照数据预处理→问题1→问题2的顺序执行
   - 确保每个步骤的输出文件存在

4. **资源要求**
   - 内存：建议8GB以上
   - 存储：至少1GB可用空间
   - GPU：用于深度学习模型（可选）

## 作者

**niupancheng** (niupancheng@163.com)

## 许可证

MIT License
