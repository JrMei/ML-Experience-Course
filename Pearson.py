import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.feature_selection import mutual_info_regression, SelectKBest

def create_squences(features: np.ndarray, target: np.ndarray, time_step: int=1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :].copy())
        X[-1][-1][7] = 0
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

def select_features_mutual_info(data, k):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(data.drop('Discharge', axis=1), data['Discharge'])
    return selector.get_support(indices=True)

def select_features_pearson(data, k):
    correlation_matrix = data.corr()
    # 确保 'Discharge' 列在 top_k_features 中
    top_k_features = correlation_matrix['Discharge'].sort_values(ascending=False).index[:k+1]
    return top_k_features.tolist()

# 读取第一个数据集
data1 = pd.read_csv("01013500_standard_normalized.csv")
data1['Year'] = data1['Date'].apply(lambda x: int(x.split('-')[0]))
data1['Month'] = data1['Date'].apply(lambda x: int(x.split('-')[1]))
data1['Day'] = data1['Date'].apply(lambda x: int(x.split('-')[2]))
data1.drop('Date', axis=1, inplace=True)
X1, y1 = create_squences(data1.to_numpy(), data1['Discharge'].to_numpy(), time_step=7)
new_data1 = pd.DataFrame(X1.reshape(X1.shape[0], -1))
new_data1['Discharge'] = y1

print("01013500\nmutual_info")
top_k_features1 = select_features_mutual_info(new_data1, 56)
print(top_k_features1)
print("pearson")
top_k_features1 = select_features_pearson(new_data1, 56)
print(top_k_features1)
print("-------------------")

# 读取第二个数据集
data2 = pd.read_csv("01047000_standard_normalized.csv")
data2['Year'] = data2['Date'].apply(lambda x: int(x.split('-')[0]))
data2['Month'] = data2['Date'].apply(lambda x: int(x.split('-')[1]))
data2['Day'] = data2['Date'].apply(lambda x: int(x.split('-')[2]))
data2.drop('Date', axis=1, inplace=True)
X2, y2 = create_squences(data2.to_numpy(), data2['Discharge'].to_numpy(), time_step=7)
new_data2 = pd.DataFrame(X2.reshape(X2.shape[0], -1))
new_data2['Discharge'] = y2

print("01047000\nmutual_info")
top_k_features2 = select_features_mutual_info(new_data2, 56)
print(top_k_features2)
print("pearson")
top_k_features2 = select_features_pearson(new_data2, 56)
print(top_k_features2)
print("-------------------")

# 读取第三个数据集
data3 = pd.read_csv("01169000_standard_normalized.csv")
data3['Year'] = data3['Date'].apply(lambda x: int(x.split('-')[0]))
data3['Month'] = data3['Date'].apply(lambda x: int(x.split('-')[1]))
data3['Day'] = data3['Date'].apply(lambda x: int(x.split('-')[2]))
data3.drop('Date', axis=1, inplace=True)
X3, y3 = create_squences(data3.to_numpy(), data3['Discharge'].to_numpy(), time_step=7)
new_data3 = pd.DataFrame(X3.reshape(X3.shape[0], -1))
new_data3['Discharge'] = y3

print("01169000\nmutual_info")
top_k_features3 = select_features_mutual_info(new_data3, 56)
print(top_k_features3)
print("pearson")
top_k_features3 = select_features_pearson(new_data3, 56)
print(top_k_features3)
print("-------------------")

# 为每个数据集生成热力图并保存
def plot_and_save_heatmap(data, top_k_features, dataset_name):
    print(f"Starting heatmap generation for {dataset_name}")
    
    # 确保 'Discharge' 在 top_k_features 中
    if 'Discharge' in top_k_features:
        columns = ['Discharge'] + [f'Lag{j+1}' for j in range(7)]
        selected_data = data[columns]
        print(f"Columns selected for {dataset_name}: {columns}")
        
        for i in range(1, 8):
            selected_data[f'Lag{i}'] = selected_data['Discharge'].shift(i)
        selected_data = selected_data.dropna()
        print(f"Data shape after shifting and dropping NaN: {selected_data.shape}")

        # 计算相关系数矩阵
        correlation_matrix = selected_data.corr()
        print(f"Correlation matrix shape: {correlation_matrix.shape}")

        # 使用Seaborn创建热力图
        plt.figure(figsize=(10, 2))
        sns.heatmap(correlation_matrix.iloc[0:1, :], annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
        plt.title(f'Pearson Correlation of Discharge with its 7-day Lags for {dataset_name}')

        # 保存热力图
        save_path = f"heatmap_{dataset_name}.png"
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
        plt.show()  # 显示热力图
        plt.close()  # 关闭图形以避免显示
    else:
        print(f"Discharge not found in top_k_features for {dataset_name}")

plot_and_save_heatmap(new_data1, top_k_features1, "01013500")
plot_and_save_heatmap(new_data2, top_k_features2, "01047000")
plot_and_save_heatmap(new_data3, top_k_features3, "01169000")