# -*- encoding: utf-8 -*-
'''
Description: Naive_Bayes_Classifier
Date: 16/5/2024
Author: Mei Jiarui
version: 1.0
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 获取原数据集并且分类，并且返回特征
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
    numList = []  
    for i in range(len(features) - 2):
        numList.append(len(feat_dic[features[i]]))

    dataSet = np.array(data_set)
    return dataSet, features

# 拉普拉斯修正
def cal_pro_lap(dataSet, index, value, classLabel, N):
    ext_data = dataSet[dataSet[:, -1] == classLabel]
    count = 0
    for data in ext_data:
        if data[index] == value:
            count += 1
    return (count + 1) / (float(len(ext_data)) + N)


# 朴素贝叶斯分类器，在这里作为基分类器
def naive_bayes_classi(dataSet, features):
    dict = {}
    for feature in features:
        index = features.index(feature)
        dict[feature] = {}
        if feature != '密度' and feature != '含糖量':
            featIList = feat_dic[feature]
            for value in featIList:
                p_true = cal_pro_lap(dataSet, index, value, '是', len(featIList))
                p_false = cal_pro_lap(dataSet, index, value, '否', len(featIList))
                dict[feature][value] = {}
                dict[feature][value]["是"] = p_true
                dict[feature][value]["否"] = p_false
        else:
            for label in ['是', '否']:
                dataExtra = dataSet[dataSet[:, -1] == label]
                extr = dataExtra[:, index].astype("float64")
                aver = extr.mean()
                var = extr.var()

                labelStr = ""
                if label == '是':
                    labelStr = '是'
                else:
                    labelStr = '否'

                dict[feature][labelStr] = {}
                dict[feature][labelStr]["aver"] = aver
                dict[feature][labelStr]["mean"] = var

    length = len(dataSet)
    classLabels = dataSet[:, -1].tolist()
    dict["好瓜"] = {}
    dict["好瓜"]['是'] = (classLabels.count('是') + 1) / (float(length) + 2)
    dict["好瓜"]['否'] = (classLabels.count('否') + 1) / (float(length) + 2)

    return dict
  
# 做一个权值的初始化  
def func_w_1(x):
    w_1=[]
    for i in range(len(x)):
        w_1.append(0.1)
    return w_1


def TrainModel(X, feature, D, verbose=True):
    """
    description:训练一个误差最低的单层决策树二分类器
    ---------
    param:
    X: 输入训练数据的特征 (size,2)
    feature: 输入训练数据的标签 (size,1)
    D: 每个样本的频率，随着集成的次数增多，每个样本不是均匀分布，D是一个(size,1)的array
    verbose: 控制是否打印切分节点的具体信息
    -------
    Returns:
    -------
    """
    m, n = X.shape # m:样本的个数，n:样本的特征数
    splitNum = 10 #每个特征为数值类型，设置均匀切分为10份
    MinError = float('inf') #记录最优模型的分类误差
    BestTree = {}
    #遍历每个特征的切分点，筛选最佳的切分点和inequal，相当于训练单层决策树模型
    for atrr in range(n):
        atrrMin = X[:,atrr].min()
        atrrMax = X[:,atrr].max()
        step = (atrrMax - atrrMin)/splitNum
        for i in range(-1,splitNum+2):
            threshold = atrrMin + i*step
            for category in ['lt','gt']:
                ErrorArray = np.zeros((m,1)) #记录预测错误的样本，正确为0，错误为1，先初始化为0
                predict =naive_bayes_classi(X,atrr,threshold,category,)
                ErrorArray[predict!=feature] = 1
                TotalError = np.dot(ErrorArray.T,D)[0,0]
                if (verbose):
                    print('第%d个特征,切分点为%.4f,误差为%.4f' %(atrr,threshold,TotalError))
                if TotalError < MinError:
                    MinError = TotalError
                    BestTree['dimen'] = atrr
                    BestTree['threshold'] = threshold
                    BestTree['inequal'] = category
                    PredictLabel = predict
    return MinError, BestTree, PredictLabel

def TrainAdaboost(features, dataset , Epochs = 40):
    """
    description: AdaBoost算法核心过程，对应西瓜书p174图8.3
    ---------
    param:
    dataset: 训练数据的特征
    feature: 训练数据的标签
    X:训练数据的特征
    y: 训练数据的标签
    Epochs: 迭代次数，对应学习器的个数
    -------
    Returns: 保存每个基学习器信息的列表
    -------
    """
    # features 标签特征
    # labelset 标签数据集
    # valueset  连续值数据集
    LearnerHistory = [] #保存基学习器的列表
    m,n = dataset.shape
    D = np.ones((m,1))*(1/m)  #初始化样本的概率分布
    AllPredict = np.zeros((m,1)) #记录所有已训练模型的集成预测结果，初始化所有元素为0

    #开始集成学习的训练过程
    for epoch in range(Epochs):
        #训练当前迭代下的基学习器
        MinError, BestTree, PredictLabel = TrainModel(dataset,features,D,verbose=False)
        #当期望误差大于0.5时候，退出循环
        if MinError > 0.5:
            break
        
        #计算当前迭代模型的权重
        alpha = 0.5*np.log((1-MinError)/max(MinError,1e-16)) #防止MinError为0导致报错

        BestTree['alpha'] = alpha
        print("第%d次迭代决策树模型第%d个特征, 切分点为%.3f, ineqal: %s,总误差为%.3f, 权重为：%.3f" % (epoch+1, BestTree['dimen'], BestTree['threshold'], BestTree['inequal'], MinError, BestTree['alpha']))
        LearnerHistory.append(BestTree)
        #更新样本的概率分布D
        expon = -alpha*(features*PredictLabel)
        D = D * np.exp(expon)
        D = D/D.sum()  # 根据样本权重公式，更新样本权重

        #计算前epoch+1个基学习器的集成算法的误差
        AllPredict += alpha*PredictLabel
        ErrorCount = np.ones((m,1)) #记录AllPredict与y中不同的预测结果
        AllPre = np.sign(AllPredict)
        print("前{}个弱分类器得到的结果:{} ".format(epoch+1, AllPre.T))
        ErrorCount[AllPre == features] = 0
        ErrorRate = ErrorCount.sum()/m
        print('分类错误率: %.3f' %ErrorRate)
        #如果错误率为0了，表示训练效果已经可以了，可不需再集成了
        if ErrorRate == 0:
            break
    
    return LearnerHistory

# 可视化，但是是后话
def visualization(X, y, EnsembleModel):
    """
    description: 可视化训练模型的结果，包含每个分类器的划分点
    ---------
    param:
    X: 输入特征
    y: 输入标签
    EnsembleModel: 集成学习训练的模型，存储在一个元素为字典的列表中
    -------
    Returns: None
    -------
    """
    m,n = X.shape
    xlimit = [X[:,0].min(),X[:,0].max()]
    ylimit = [X[:,1].min(),X[:,1].max()]
    GoodMelon = []
    BadMelon = []
    for i in range(m):
        if y[i,-1] > 0:
            GoodMelon.append(X[i,:].tolist())
        else:
            BadMelon.append(X[i,:].tolist())
    GoodMelon = np.array(GoodMelon)
    BadMelon = np.array(BadMelon)

    plt.rc('font',family='Times New Roman')
    plt.scatter(GoodMelon[:,0],GoodMelon[:,1],s=30,c='red',marker='o',alpha=0.5,label='Good')
    plt.scatter(BadMelon[:,0],BadMelon[:,1],s=30,c='blue',marker='x',alpha=0.5,label='Bad')

    for baseleaner in EnsembleModel:
        print(baseleaner)
        if baseleaner['dimen'] == 0:
            plt.plot([baseleaner['threshold'],baseleaner['threshold']],ylimit,linestyle=':')
        else:
            plt.plot(xlimit,[baseleaner['threshold'],baseleaner['threshold']],linestyle=':')
    
    plt.legend()
    plt.xlabel('density')
    plt.ylabel('Sugar content')
    plt.show()
    return None

if __name__ == '__main__':
    dataset ,features= get_data() # 读入数据集
    dict = naive_bayes_classi(dataSet=dataset,features=features) # 得到朴素贝叶斯分类器分类后得到的结果
    m,n = dataset.shape()
    # print(SingleTree(X,0,0.3,'lt'))
    # D = np.ones((m,1))*(1/m)
    # MinError, BestTree, PredictLabel = TrainModel(X,y,D)
    # print(BestTree)
    # print(MinError)

    # model = TrainAdaboost(X, y, 40) #训练AdaBoost模型
    # visualization(X,y,model)   