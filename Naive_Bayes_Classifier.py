import numpy as np
import pandas as pd
import sklearn
import sklearn.naive_bayes

# 设定通用变量
node_level = -1
feature_list = [[] for _ in range(3)]
value_list = [[] for _ in range(3)]
total_labels = [['青绿', '乌黑', '浅白'],
                ['蜷缩', '稍蜷', '硬挺'],
                ['清脆', '浊响', '沉闷'],
                ['稍糊', '清晰', '模糊'],
                ['凹陷', '稍凹', '平坦'],
                ['硬滑', '软粘']]
labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '含糖量', '密度', '好坏']

# 特征字典
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}

test_list=['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460] #实验要求数据集

# 获取数据集
def get_data():
    original_dataset = list(np.array(pd.read_csv('/Users/meijiarui/Desktop/机器学习实验/data/watermelon3_0_Ch.csv')))
    labelset = [] # 存储处理后的数据集，包含特征和标签
    valueset = [] # 存储数据集中的连续值，用于后续离散化处理

    # 遍历数据集中的每一行
    for i in range(len(original_dataset)):
        temp_data = [] # 临时存储当前行的特征值
        temp_value = [] # 临时存储当前行的连续值
        # 遍历当前行的每个特征值
        for j in range(1, len(original_dataset[i]) - 1):
            feature_value = original_dataset[i][j]# 判断特征值的索引，如果是第7或第8个特征，则将其视为连续值
            if j == 7 or j == 8:
                temp_value.append(feature_value)
            else:
                temp_data.append(feature_value)
        # 将当前行的最后一个特征值（标签）添加到temp_data列表
        temp_data.append(original_dataset[i][9])
        temp_value.append(original_dataset[i][9])
        labelset.append(temp_data)
        valueset.append(temp_value)

    return labelset, valueset

# 拉普拉斯修正
def cal_pro_Lap(dataset, index , value , classLabel, N):
    extrData = dataset[dataset[:, -1] == classLabel] #提取符合的数据集label
    cnt = 0 # 记录次数
    for data in extrData: #遍历数据
        if data[index] == value:
            cnt += 1 
    return (cnt + 1)/float(len(extrData)+N) # 拉普拉斯修正返回函数

# 连续属性概率密度
def cal_contin_prob(sigma,Xi,mu):
    # sigma:方差
    # mu : 期望
    # Xi : xi样本
    return np.exp(-(Xi-mu)**2/(2*(sigma**2)))/(np.sqrt(2*np.pi)*sigma)

# 朴素贝叶斯
def naive_bayes_class(labelset,valueset,features):
    
    dict={}
    for feature in features:
        index = features.index(feature)
        dict[feature]={}  # 建立贝叶斯概率的元素字典，用于储存不同判别下的概率值
        featu_list = feature_list[feature]  # 可能的取值
        # 记录好瓜坏瓜概率计算
        for val in featu_list:
            P_true_prob = cal_pro_Lap(labelset, index, val, '1',len(featu_list))
            P_false_prob = cal_pro_Lap(labelset, index, val, '0',len(featu_list))
            dict[feature][val] = {}
            dict[feature][val]['是'] = P_true_prob # 记录情况下为好瓜的概率
            dict[feature][val]['否'] = P_false_prob # 记录情况下为坏瓜的概率
        # 计算连续数据值的方差和均值
        for label_tof in range['是','否']:
            data_Ext = valueset[valueset[:,-1] == label_tof]
            ext = float(data_Ext[:,index])
            sigma = np.var(ext) # 计算方差
            mu = np.mean(ext) # 计算期望
            
            label_str =''
            dict[feature][label_str] = {} # 开个特性数值字典
            dict[feature][label_str]['方差'] = sigma
            dict[feature][label_str]['期望'] = mu           
            
    length = len(labelset) + len(valueset)
    label_class = labelset[:,-1].tolist()
    dict['好瓜']={}
    dict['好瓜']['是'] = (label_class.count('是') + 1) / (float(length) + 2) 
    dict['好瓜']['否'] = (label_class.count('否') + 1) / (float(length) + 2)
    
    return dict


# 贝叶斯预测数据
def bayes_predic(test_data, features, dic):
    prob_good = dic['好瓜']['是']
    prob_bad = dic['好瓜']['否']
    
    for feature in features:
        if feature in ['密度', '含糖量']:
            # 计算连续特征的概率密度并乘以概率
            sigma = dic[feature][label_str]['方差']
            mu = dic[feature][label_str]['期望']
            prob_density = cal_contin_prob(sigma, test_data[feature], mu)
            if label_str == '是':
                prob_good *= prob_density
            else:
                prob_bad *= prob_density
        else:
            index = features.index(feature)
            label_str = '是' if test_data[index] == '1' else '否'
            prob_good *= dic[feature][test_data[index]][label_str]
            prob_bad *= dic[feature][test_data[index]]['否']
    
    # 返回较高概率的类别
    return '是' if prob_good > prob_bad else '否'

if __name__ == '__main__':
    labelset, valueset = get_data()  # 获取数据集
    features = labelset[0]  # 假设所有数据集的特征顺序相同
    
    bayes_dict = naive_bayes_class(labelset, valueset, features)  # 计算朴素贝叶斯概率
    test_data = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑'}
    test_data['含糖量'] = 0.697  # 假设含糖量为 0.697
    test_data['密度'] = 0.460  # 假设密度为 0.460
    
    prediction = bayes_predic(test_data, features, bayes_dict)  # 使用朴素贝叶斯进行预测
    print(f"预测结果: {prediction}")