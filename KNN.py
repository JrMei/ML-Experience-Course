# 题目：编程实现K近邻分类器，在此txt数据集上比较其分类边界与决策树分类边界的异同
"""""
Date: 6.16.2024
Ver:1
Coder: Mei
"""""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 设定通用变量
labels = [ '密度', '含糖量', '好坏']

def read_txt(file_name):
    data = [] # 数据集
    with open(file_name, 'r') as file:
        origin_labels = file.readline()
        for row in file:
            tmp_list = row.split()
            data.append([float(item) for item in tmp_list])
    return np.array(data)

file_path = '/Users/meijiarui/Desktop/机器学习实验/data/watermelon3_0a_Ch.txt'
valueset = read_txt(file_path)


# 将数据集加载并分割为特征和标签
X = np.array(valueset[:,:-1])
y = np.array(valueset[:,-1].astype(int))

# 可视化决策边界的辅助函数
def plot_decision_boundaries(X, y, classifier, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

# KNN算法
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
plot_decision_boundaries(X, y, knn, "KNN-Decision Boundary")


# 决策树算法
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(X, y)
plot_decision_boundaries(X, y, tree, "Tree-Decision Boundary")
