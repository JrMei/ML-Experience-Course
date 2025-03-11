import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成两类样本
def generate_binary_data(n_samples=100, n_features=2, random_state=42):
    np.random.seed(random_state)
    # 生成第一类样本
    X1 = np.random.randn(n_samples//2, n_features) - [2, 2]
    # 生成第二类样本
    X2 = np.random.randn(n_samples//2, n_features) + [2, 2]
    # 合并两类样本
    X = np.vstack([X1, X2])
    # 创建对应的标签数组，确保长度与样本数量一致
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y

# 训练逻辑回归模型
def train_logistic_regression(X, y):
    log_reg = LogisticRegression(max_iter=1000) #调用逻辑回归函数
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 将数据集x和标签y分割为训练集和测试集
    log_reg.fit(X_train, y_train) #使用训练集数据训练逻辑回归模型
    y_pred = log_reg.predict(X_test) #训练后的逻辑回归模型对测试集进行预测
    accuracy = accuracy_score(y_test, y_pred) #计算预测准确率
    return log_reg, X, y, accuracy

# 训练LDA模型
def train_lda(X, y):
    lda = LinearDiscriminantAnalysis() #调用sklearn库中的线性判别分析模型函数
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 将数据集x和标签y分割为训练集和测试集
    lda.fit(X_train, y_train) #使用fit函数训练模型
    X_lda = lda.transform(X_train) #使用lda模型对训练集数据进行降维处理，并得到降维数据
    y_pred = lda.predict(X_test) #使用训练后的lda模型对数据进行预测
    accuracy = accuracy_score(y_test, y_pred) #计算准确率
    return lda, X_lda, y_train, accuracy

# 绘制决策边界
def plot_decision_boundaries(X, y, model, title):
    # 创建网格以绘制模型的决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #获取特征中的第一个特征的最小值和最大值创建网络
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 #获取特征中的第二个特征最小值和最大值创建网络
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) #生成等差数组
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) #将预测结果Z重新整形为xx网格形状
    plt.contourf(xx, yy, Z, alpha=0.4) #二分法分割
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k') #绘制数据集点
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show() # 实验结果图

# 主程序
if __name__ == "__main__":
    # 生成数据
    X, y = generate_binary_data()

    # 训练逻辑回归模型并分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg, _, _, accuracy_lr = train_logistic_regression(X_train, y_train)
    
    #打印X与y的数值
    print(X, y)
    
    # 绘制逻辑回归决策边界
    plot_decision_boundaries(X_train, y_train, log_reg, 'Logistic Regression Decision Boundaries')
    
    # 绘制LDA决策边界
    lda, _, _, accuracy_lda = train_lda(X, y)
    plot_decision_boundaries(X_train, y_train, lda, 'LDA Decision Boundaries')