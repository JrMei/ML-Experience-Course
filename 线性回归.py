import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#思路是随机生成x轴坐标来产生数据集，并且加上噪声使得数据点并不处于同一条直线上
#在参量中定义出斜率（true_slope)和截距(true_intercept)，即线性模型ax+b中的a与b
def generate_noisy_data(num_samples, true_slope, true_intercept, noise_std):

    x = np.random.uniform(-10, 10, num_samples) #随机给定噪声数据的x轴数据
    y = true_slope * x + true_intercept + np.random.normal(0, noise_std, num_samples)
    print(x,y)
    return x, y

def fit_linear_model(x, y):
    model = LinearRegression() #模型调用线性回归
    model.fit(x.reshape(-1, 1), y) #训练形成模型
    return model

def plot_data_and_fit(x, y, model):

    plt.scatter(x, y, color='blue', label='Data points')
    x_fit = np.linspace(min(x), max(x), 100) #特征数组的计算
    y_fit = model.coef_[0] * x_fit + model.intercept_ #此处model为拟合后的回归模型
    plt.plot(x_fit, y_fit, color='red', label=f'Fitted line y={model.coef_[0]:.2f}x+{model.intercept_:.2f}')
    plt.legend() 
    plt.show() #绘图

# 代码生成主程序
if __name__ == "__main__":
    num_samples = 50 #定义50个样本个数
    true_slope = 2 
    true_intercept = 3
    noise_std = 5

    # 生成数据
    x, y = generate_noisy_data(num_samples, true_slope, true_intercept, noise_std)

    # 拟合模型
    model = fit_linear_model(x, y)

    # 绘制数据和拟合直线
    plot_data_and_fit(x, y, model)