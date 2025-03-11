'''
Description: Naive_Bayes_Classifier & Laplace Cor
Date: 16/5/2024
Author: Mei Jiarui
version: 1.0

'''
import numpy as np
import pandas as pd 

# 特征字典
feat_dic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}

# 特征标签
features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']

# 获取原始数据集，并根据需要进行数据预处理
def get_data():
    original_dataset = list(np.array(pd.read_csv('/Users/meijiarui/Desktop/机器学习实验/data/watermelon3_0_Ch.csv')))
    data_set = []

    # 遍历数据集中的每一行
    for i in range(len(original_dataset)):
        temp_dataset = [] # 临时存储当前行的特征值
        # 遍历当前行的每个特征值
        for j in range(1, len(original_dataset[i])):
            feature_value = original_dataset[i][j]
            temp_dataset.append(feature_value)
        # 将当前行的最后一个特征值（标签）添加到temp_data列表
        data_set.append(temp_dataset)
        
    # 每种特征的属性个数
    numList = []  # 创建属性的list
    for i in range(len(features) - 2):
        numList.append(len(feat_dic[features[i]]))

    dataSet = np.array(data_set) # 数据集转换为array
    return dataSet, features

# 拉普拉斯修正操作
def cal_pro_lap(dataSet, index, value, classLabel, N):
    ext_data = dataSet[dataSet[:, -1] == classLabel] # 这里是获取数据集最后一列的的值
    # 以上还进行了一个与所需要的特征所比较的过程
    count = 0
    for data in ext_data:
        if data[index] == value:
            count += 1
    return (count + 1) / (float(len(ext_data)) + N)

# 计算连续概率
def cal_contin_prob(mean, var, xi):
    return np.exp(-((float(xi) - mean) ** 2) / (2 * var)) / (np.sqrt(2 * np.pi * var))

# 朴素贝叶斯分类器
def naive_bayes_classi(dataSet, features):
    dict = {} # 创建一个用于储存贝叶斯分类器中概率计算的值的字典
    for feature in features: # 遍历所有特征
        index = features.index(feature)
        dict[feature] = {} # 作为特定特征标签下的概率计算结果保存
        if feature != '密度' and feature != '含糖量': # 当不是连续性的概率值计算时的情况
            featIList = feat_dic[feature]
            for value in featIList:
                # 以下计算在拉普拉斯修正下的是否为好瓜的判断的概率值的计算
                p_true = cal_pro_lap(dataSet, index, value, '是', len(featIList))
                p_false = cal_pro_lap(dataSet, index, value, '否', len(featIList))
                # 在本函数开头创建的字典中保存对应标签和瓜的好坏下的概率值
                dict[feature][value] = {}
                dict[feature][value]["是"] = p_true
                dict[feature][value]["否"] = p_false
        else: # 这里是”密度“、”含糖量“两个连续值（浮点数）的计算情况
              # 但是不同于前类，这里计算的是方差和均值，这是为了后续的概率计算
            for label in ['是', '否']:
                dataExtra = dataSet[dataSet[:, -1] == label]
                extr = dataExtra[:, index].astype("float64")
                aver = np.average(extr) # 计算均值 
                var = np.var(extr) # 计算方差
        
                dict[feature][label] = {}
                dict[feature][label]["aver"] = aver # 均值字典
                dict[feature][label]["mean"] = var # 方差字典

    length = len(dataSet)
    classLabels = dataSet[:, -1].tolist()
    dict["好瓜"] = {} # 开一个字典储存朴素贝叶斯计算以后得到的好坏瓜的两类最终概率P
    # 以下使用贝叶斯公式来计算对应的最终要得到的概率的计算值
    dict["好瓜"]['是'] = (classLabels.count('是') + 1) / (float(length) + 2)
    dict["好瓜"]['否'] = (classLabels.count('否') + 1) / (float(length) + 2)

    return dict


# 判断函数
def judge(P_good,P_bad):
    if P_good > P_bad:
        ret_class = "好瓜"
    else:
        ret_class = "坏瓜"
    return ret_class

# 预测数据和数据类型
def predict(data, features, bayesDis):
    P_good = bayesDis['好瓜']['是']
    P_bad = bayesDis['好瓜']['否']
    for feature in features:
        index = features.index(feature)
        if feature != '密度' and feature != '含糖量':
            P_good *= bayesDis[feature][data[index]]['是']
            P_bad *= bayesDis[feature][data[index]]['否']
        else:
            # 调用连续值的计算公式 
            P_good *= cal_contin_prob(bayesDis[feature]['是']['aver'],
                              bayesDis[feature]['是']['mean'],
                              data[index])
            P_bad *= cal_contin_prob(bayesDis[feature]['否']['aver'],
                              bayesDis[feature]['否']['mean'],
                              data[index])

    # 类别判断
    ret_class = ""
    ret_class = judge(P_good, P_bad)  # 调用判断函数
    return P_good, P_bad, ret_class

if __name__ == '__main__':  
    dataSet, features = get_data() 
    dic = naive_bayes_classi(dataSet, features)
    # print(dic) #有需要的话打印字典 这里是为了检查代码
    p_true, p_false, result = predict(dataSet[0], features, dic)
    print(f"好瓜概率 = {p_true}")
    print(f"坏瓜概率 = {p_false}")
    print(f"结果 = {result}")

# 版本可用
