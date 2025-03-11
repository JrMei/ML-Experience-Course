import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 生成带有随机噪声的原始数据集
X, Y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.6)

# 分割数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 训练支持向量机模型
linear_svm_model = LinearSVC().fit(X_train, Y_train)

# 预测测试集
Y_pred = linear_svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印分类报告
print(classification_report(Y_test, Y_pred))

# 计算图形边界
l, r, h = X.min() - 1, X.max() + 1, 0.02  # 使用整个数据集的范围来绘制边界
x_values = np.arange(l, r, h)
y_values = np.arange(l, r, h)
X1, X2 = np.meshgrid(x_values, y_values)
grid = np.c_[X1.ravel(), X2.ravel()]

# 预测网格点的类别
Y_grid = linear_svm_model.predict(grid)

# 绘制决策边界
plt.figure("SVM Classifier", facecolor="lightgray")
plt.title("SVM Classifier with Random Noise", fontsize=14)

plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.tick_params(labelsize=10)
plt.pcolormesh(X1, X2, Y_grid.reshape(X1.shape), cmap="gray")

# 绘制训练集和测试集数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=50, edgecolor='k', label='Training data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=50, edgecolor='k', label='Test data')

plt.legend()
plt.show()
