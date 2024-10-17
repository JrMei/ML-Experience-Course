import pandas as pd
import matplotlib.pyplot as plt

# 假设有三个CSV文件，分别是 dataset1.csv, dataset2.csv, dataset3.csv
files = ['01013500.csv', '01047000.csv', '01169000.csv']

# 定义空的列表，用于存储每个文件的相同列
dataframes = []
data_units = []

# 定义需要绘制箱线图的列
columns_to_plot = ['Discharge', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']
unit = ['m^2', 's/d', 'mm', 'W/m^2', 'kg/m^2', 'C', 'C', 'Pa']

# 遍历每个CSV文件并读取相关列
for i, file in enumerate(files):
    df = pd.read_csv(file)
    df['Source'] = f'Dataset {i+1}'  # 为每个数据集添加来源列，标注数据集名称
    dataframes.append(df[['Discharge', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp', 'Source']])  # 只保留所需列和 Source 列

# 将所有数据集的数据合并到一个DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# 创建图像和子图, 这里使用 2 行 4 列的布局
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 行 4 列，图像大小为 20x10

# 将 axes 从二维数组转换为一维数组，方便遍历
axes = axes.flatten()

# 遍历每个要绘制的列，并在不同的子图中绘制箱线图
for i, column in enumerate(columns_to_plot):
    combined_df.boxplot(column=column, by='Source', ax=axes[i], grid=False)  # 在不同的子图上绘制
    axes[i].set_title(f'{column} ')  # 设置子图标题
    axes[i].set_xlabel('Dataset')  # 设置X轴标签
    axes[i].set_ylabel(f'{column} / {unit[i]}')  # 设置Y轴标签

# 设置总体标题，并去除自动生成的子标题
plt.suptitle('')
plt.tight_layout()  # 自动调整子图布局
plt.show()
