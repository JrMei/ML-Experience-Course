import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

# Step 4: 训练随机森林模型，并添加训练时间计时
def train_random_forest(X_train, y_train):
    """
    使用随机森林进行训练，并通过GridSearchCV调整超参数。计时功能用于记录训练时间。
    """
    rf = RandomForestRegressor()

    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 400],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt']
    }

    # 使用时间序列交叉验证
    grid_search = GridSearchCV(rf, param_grid, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')

    # 添加训练时间计时
    print("Model Training Starting...")
    start_time = time.time()  # 开始计时
    try:
        grid_search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time  # 计算训练时间
        print(f"Model Training Success! Time Taken: {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Model Training Failed: {e}")
        return None, None
    
    return grid_search.best_estimator_, elapsed_time

# Step 5: 评估模型性能，增加score计算
def evaluate_model(model, X_test, y_test):
    """
    对测试集进行预测，并计算均方误差(MSE)和决定系数(R² score)。
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)  # 计算 R² score
    return mse, r2, predictions

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
    
    # 训练随机森林模型并记录训练时间
    model, train_time = train_random_forest(X_train, y_train)
    
    if model is None:
        print("Training failed, unable to continue and visualize")
        return
    
    # 评估模型
    mse, r2, predictions = evaluate_model(model, X_test, y_test)
    
    # 输出评估指标和训练时间
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R2 Score: {r2}")
    print(f"Training Time: {train_time:.2f} seconds")
    
    # 可视化预测值和残差
    plot_predictions(y_test, predictions)
    plot_residuals(y_test, predictions)

    return model, predictions

if __name__ == "__main__":
    file_path = '01169000_minmax_normalized.csv'  # 替换为实际数据文件路径
    # 运行流水线并输出结果
    model, predictions = run_pipeline_with_visualization(file_path)