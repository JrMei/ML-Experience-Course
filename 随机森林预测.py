import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Step 1: 加载经过正则化后的数据
def load_normalized_data(file_path):
    """
    从CSV文件中加载经过正则化后的数据，确保日期列被正确解析。
    """
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

# Step 2: 按照时间顺序划分数据为训练集和测试集
def split_data(data):
    """
    根据年份划分数据。前4年为训练集，第5年为测试集。
    """
    train_data = data[data.index.year < data.index.year.max()]
    test_data = data[data.index.year == data.index.year.max()]
    return train_data, test_data

# Step 3: 构建特征与目标数据
def prepare_features_and_target(data, n_lags=5):
    """
    构建用于模型的特征和目标数据。使用过去n_lags天的数据预测未来5天的径流。
    """
    X = []
    y = []
    for i in range(n_lags, len(data)-4):  # 确保预测5天
        X.append(data.iloc[i - n_lags:i].values.flatten())  # 将三维特征矩阵转换为二维
        y.append(data['Discharge'].iloc[i:i + 5].values)  # 预测未来5天
    return np.array(X), np.array(y)

# Step 4: 训练随机森林模型
def train_random_forest(X_train, y_train):
    """
    使用随机森林进行训练，并通过GridSearchCV调整超参数。
    """
    rf = RandomForestRegressor() # 可以直接支持多输出
    #multi_rf = MultiOutputRegressor(rf) # 可以不使用
    
    # 定义超参数搜索空间
    # 修改param_grid,修改GridSearchCV随机森林网络搜索
    """
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [10, 20, None],
        'estimator__min_samples_split': [2, 5, 10]
    }
    """
    # 设置随机森林回归模型的参数
    # 随机森林自己带有输出
    # 修改参数使得结果符合之前的部分
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 400],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt']
    }
    
    # 使用时间序列交叉验证
    # grid_search = GridSearchCV(multi_rf, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    grid_search = GridSearchCV(rf, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    
    # 添加进度提示
    print("Model Training Starting...")
    try:
        grid_search.fit(X_train, y_train)
        print("Model Training Success !")
    except Exception as e:
        print(f"Model Training Failed: {e}")
        return None
    
    return grid_search.best_estimator_

# Step 5: 评估模型性能
def evaluate_model(model, X_test, y_test):
    """
    对测试集进行预测，并计算均方误差(MSE)。
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Step 6: 绘制实际值与预测值的对比图
def plot_predictions(actuals, predictions, title="Contrast of Actual value and Predict value"):
    """
    绘制时间序列图，比较实际值与模型预测值。
    """
    plt.figure(figsize=(14, 6))
    days = np.arange(1, len(actuals) + 1)

    for i in range(predictions.shape[1]):  # 绘制未来5天的预测结果
        plt.plot(days, predictions[:, i], label=f'Predict Day{i+1}', linestyle='--')

    plt.plot(days, actuals[:, 0], label='Actual Discharge', color='black', linewidth=2)

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 7: 绘制残差图
def plot_residuals(actuals, predictions, title="Residual (predicted value - actual value)"):
    """
    绘制残差图，展示预测值与实际值之间的差异。
    """
    residuals = predictions - actuals[:, 0].reshape(-1, 1)  # 计算残差
    plt.figure(figsize=(14, 6))
    days = np.arange(1, len(actuals) + 1)

    for i in range(residuals.shape[1]):
        plt.plot(days, residuals[:, i], label=f'Residual Day{i+1}', linestyle='--')

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid(True)
    plt.show()

# 完整流程，包括可视化
def run_pipeline_with_visualization(file_path):
    """
    执行完整流程：数据加载、预处理、训练模型、评估模型、并进行可视化。
    """
    data = load_normalized_data(file_path)
    
    # 划分训练集和测试集
    train_data, test_data = split_data(data)
    
    # 准备特征和目标数据
    X_train, y_train = prepare_features_and_target(train_data)
    X_test, y_test = prepare_features_and_target(test_data)
    
    # 训练随机森林模型
    model = train_random_forest(X_train, y_train)
    
    if model is None:
        print("Training failed , unable to continue and visualize")
        return
    
    # 评估模型
    mse, predictions = evaluate_model(model, X_test, y_test)
    
    print(f"MSE: {mse}")
    
    # 可视化预测值和残差
    plot_predictions(y_test, predictions)
    plot_residuals(y_test, predictions)

    return model, predictions

if __name__ == "__main__":
    file_path = '01169000_standard_normalized.csv'  # 替换为实际数据文件路径
    run_pipeline_with_visualization(file_path)