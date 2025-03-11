import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

# 设置通用变量
labels = ['密度', '含糖量', '好坏']

# 定义读取txt文件的函数
def read_txt(file_name):
    data = []
    with open(file_name, 'r') as file:
        origin_labels = file.readline()  # 读取并忽略第一行的标签
        for row in file:
            tmp_list = row.split()  # 按空格分割每一行的数据
            data.append([float(item) for item in tmp_list])  # 将每一行的数据转换为float并添加到data列表中
    return np.array(data)  # 转换为numpy数组

# 文件路径
file_path = '/Users/meijiarui/Desktop/机器学习实验/data/watermelon3_0a_Ch.txt'

# 读取数据
valueset = read_txt(file_path)

# 将数据集分割为特征和标签
X = valueset[:, :-1]  # 所有行，除最后一列之外的列作为特征
y = valueset[:, -1].astype(int)  # 最后一列作为标签，并转换为整数类型

# 将特征和标签组合为一个DataFrame
columns = ['density', 'sugar_rate', 'label']
data_set = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=columns)

# 定义KMeans类
class KMeans:
    def __init__(self, k, data, max_iterations, tolerance):
        self.k = k
        self.data = data
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def calculate_distance(self, point1, point2):
        # 计算两点之间的欧氏距离
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def fit(self):
        start_time = time.perf_counter()  # 记录开始时间
        centroids = random.sample(self.data, self.k)  # 随机选择k个初始质心
        initial_centroids = centroids.copy()  # 复制初始质心

        # 绘制初始质心
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], s=100, color='red', marker='x')

        iterations = 0  # 初始化迭代次数
        clusters = [[centroid] for centroid in centroids]  # 初始化簇

        while iterations < self.max_iterations:
            change_flag = False  # 标记质心是否改变
            for sample in self.data:
                distances = [self.calculate_distance(sample, centroid) for centroid in centroids]
                closest_centroid_index = distances.index(min(distances))  # 找到最近质心的索引
                clusters[closest_centroid_index].append(sample)  # 将样本添加到最近质心的簇中

            new_centroids = []
            for cluster, centroid in zip(clusters, centroids):
                cluster_array = np.array(cluster)
                new_centroid = np.mean(cluster_array, axis=0).tolist()  # 计算新质心
                # 判断质心是否改变
                if all(np.abs((np.array(new_centroid) - np.array(centroid)) / np.array(centroid)) < self.tolerance):
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(new_centroid)
                    change_flag = True

            if not change_flag:
                break

            centroids = new_centroids
            clusters = [[centroid] for centroid in centroids]  # 重置簇
            iterations += 1  # 增加迭代次数

        end_time = time.perf_counter()  # 记录结束时间
        print(f'{self.k} initial centroids{initial_centroids}')
        print(f'Total_iterations{iterations}')
        print(f'Time{end_time - start_time:.2f}s')

        # 绘制聚类结果
        for cluster, centroid in zip(clusters, centroids):
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            plt.scatter(x, y, marker='o', label=f'Cluster {clusters.index(cluster) + 1}')
            # 绘制数据点与质心的连线
            for point in cluster:
                plt.plot([centroid[0], point[0]], [centroid[1], point[1]], 'k--', linewidth=0.5)

        plt.xlabel('Density')
        plt.ylabel('Sugar Rate')
        plt.legend(loc='upper left')
        plt.show()

for k in [2,3,4]:
  k_means = KMeans(k, data_set[['density', 'sugar_rate']].values.tolist(), 1000, 1e-7)
  k_means.fit()