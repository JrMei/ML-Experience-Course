import pandas as pd
import numpy as np
from math import *

def load_data(path):
    # 加载数据集
    original_dataset = pd.read_csv(path) # 原始数据集
    
    # 初始化列表
    true_data = [] # 真类数据集
    false_data = [] # 假类数据集
    val_count = [] # 每个特征的唯一值数量
    true_weights = [] # 真类数据集的权重
    false_weights = [] # 假类数据集的权重

    # 将数据集分为真类和假类
    for i in range(len(original_dataset)):
        if original_dataset.iloc[i, -1] == '是':
            true_weights.append(1 / len(original_dataset))
        else:
            false_weights.append(1 / len(original_dataset))

    # 将权重转换为numpy数组，便于以后计算
    true_weights = np.array(true_weights)
    false_weights = np.array(false_weights)

    # 将数据分为真类和假类数据集
    for i in range(len(original_dataset)):
        data = list(original_dataset.iloc[i, 1:])
        if original_dataset.iloc[i, -1] == '是':
            true_data.append(data)
        else:
            false_data.append(data)

    # 计算每个特征的唯一值数量
    for i in range(original_dataset.shape[1] - 1):
        if isinstance(original_dataset.iloc[0, i], str):
            val_count.append(len(set(original_dataset.iloc[:, i])))
        else:
            val_count.append(0)  # 对于数值特征，特征值数量不适用

    return true_data, false_data, val_count, true_weights, false_weights, original_dataset

# 贝叶斯作为基分类器
def bayes_prob_cal(test_list, dataset, initial_rate, val_count, weights, total_len):
    rate = initial_rate
    for i in range(len(test_list) - 1):
        value = test_list[i]
        if isinstance(value, str):
            # 对于离散特征，计算条件概率
            # 这里调用isinstance函数来判断
            count = sum(weights[j] * total_len for j in range(len(dataset)) if dataset[j][i] == value)
            p_temp = (count + 1) / (len(dataset) + val_count[i]) # 这里的p_temp代表了概率值会由于权重而不断更新
        else:
            # 对于连续特征，计算高斯分布概率密度
            value_list = np.array([float(data[i]) for data in dataset])
            mean = np.mean(value_list) # 计算数组均值
            std = np.std(value_list, ddof=1) #计算value_list的标准差
            #参数ddof=1表示在计算方差时使用“样本方差”公式，即分母是N-1，其中N是样本数量
            p_temp = (1 / (sqrt(2 * pi) * std)) * exp(-((value - mean) ** 2) / (2 * (std ** 2)))
            # 计算概率密度函数
            # 这部分和上面的概率计算同理，这里计算的是连续特征的概率
        
        rate *= p_temp # 测试样本的概率值
    return rate


def AdaBoost(true_data, false_data, val_count, true_weights, false_weights, inte=10):
    all_set = true_data + false_data #真类数据集和假类数据集合并到一个单一的列表中
    total_len = len(all_set) # 计算合并后数据集的长度，即总样本数
    true_rate_init = (len(true_data) + 1) / (total_len + 2) #计算初始时真类的先验概率，使用了拉普拉斯修正
    false_rate_init = (len(false_data) + 1) / (total_len + 2)
    weights = np.append(true_weights, false_weights)
    # 将真类数据集的权重和假类数据集的权重合并到一个数组weights中这些权重在 AdaBoost 的每一轮迭代中都会更新
    alpha = []# 初始化一个空列表 alpha，用于存储每一轮迭代后的权重更新因子
    true_rate_set = np.zeros((inte, total_len))#用于存储每一轮迭代中每个样本作为真类的概率
    false_rate_set = np.zeros((inte, total_len))#用于存储每一轮迭代中每个样本作为假类的概率
    
    for t in range(inte):
        error = 0
        results = [] # 结果字典储存结果
        
        # 对每个样本进行预测
        for i in range(total_len):
            test_list = all_set[i]
            true_rate = bayes_prob_cal(test_list, true_data, true_rate_init, val_count, true_weights, total_len)
            false_rate = bayes_prob_cal(test_list, false_data, false_rate_init, val_count, false_weights, total_len)

            true_rate_set[t, i] = true_rate
            false_rate_set[t, i] = false_rate

            # 根据概率大小做出预测
            prediction = '是' if true_rate > false_rate else '否'
            results.append(prediction)  # 更新结果

            # 计算误差
            if prediction != test_list[-1]:
                error += weights[i]
            # 如果样本的预测结果prediction与样本的真实标签test_list[-1]不一致，则更新error变量
            # 这里使用了权重weights[i] 来计算加权误差

        if error == 0:
            error = 1e-10  # 避免除以零
        alpha_t = 0.5 * log((1 - error) / error) # 根据公式迭代因子alpha
        alpha.append(alpha_t)

        # 更新权重
        for i in range(total_len):
            if results[i] != all_set[i][-1]:
                weights[i] *= exp(alpha_t)
            else:
                weights[i] *= exp(-alpha_t)

        # 归一化权重
        weights /= np.sum(weights)
        true_weights = weights[:len(true_weights)]
        false_weights = weights[len(true_weights):]

        print(f"第 {t + 1} 次迭代：")
        print(f"误差：{error:.4f}, Alpha:{alpha_t:.4f}")
        print(f"预测结果：{results}\n")

    final_predictions = [] # 最后的预测结果
    for i in range(total_len):
        #使用sum函数和数组乘法计算所有迭代中，该样本作为真（或假）的累积概率，存储在true_res中
        true_res = np.sum(true_rate_set[:, i] * alpha)
        false_res = np.sum(false_rate_set[:, i] * alpha)
        final_prediction = '是' if true_res > false_res else '否'
        final_predictions.append((all_set[i], final_prediction))
    
    return final_predictions

if __name__ == '__main__':
    path = '/Users/meijiarui/Desktop/机器学习实验/data/watermelon3_0_Ch.csv'
    true_data, false_data, val_count, true_weights, false_weights, original_dataset = load_data(path)
    final_predictions = AdaBoost(true_data, false_data, val_count, true_weights, false_weights)
    print("最终预测结果：")
    for data_point, prediction in final_predictions:
        if prediction == '是':
            print(f"样本 {data_point} → 预测结果：这是一个好瓜")
        else:
            print(f"样本 {data_point} → 预测结果：这是一个坏瓜")
    
