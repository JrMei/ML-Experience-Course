from math import log
import numpy as np
import pandas as pd

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

def get_data():
    original_dataset = list(np.array(pd.read_csv('/Users/meijiarui/Desktop/机器学习实验一/data/watermelon3_0_Ch.csv')))
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
        labelset.append(temp_data)
        valueset.append(temp_value)

    return labelset, valueset

# 计算给定数据集的熵。
def calc_entropy(all_count, true_count):
    entropy = 0
     # 如果数据集中既有正例又有负例，则计算熵
    if all_count - true_count > 0 and true_count != 0:
        entropy -= (true_count / all_count) * log(true_count / all_count, 2) + ((all_count - true_count) / all_count) * log((all_count - true_count) / all_count, 2)
    return entropy

# 计算数据集的基础熵。统计数据集中正例（标签为'是'）的数量，并基于此计算熵。
def calc_base_entropy(dataset):
    true_count = sum(1 for example in dataset if example[-1] == '是')
    return calc_entropy(len(dataset), true_count)

# 将连续值转换为离散标签的函数。对连续值进行排序，并计算每个分割点的熵，以找到最佳分割点。
def value_to_label(labelset, valueset):
    base_entropy = calc_base_entropy(valueset)
    t_ent = [0] * 2 # 存储每个分割点的熵值
    m_value = [] # 存储离散化后的密度值
    t_value = [] # 存储离散化后的含糖量值
    t_m_div = 0  # 初始化密度分割点
    t_t_div = 0  # 初始化含糖量分割点

    # 对valueset中的连续值进行排序
    for i in range(len(valueset)):
        m_value.append(valueset[i][0])
        t_value.append(valueset[i][1])

    m_value.sort()  # 密度值排序
    t_value.sort() # 含糖量值排序

    # 遍历排序后的连续值，寻找最佳分割点
    for i in range(len(valueset) - 1):
        count = [0] * 4   # 存储每个区间的样本数量
        true_count = [0] * 4  # 存储每个区间的正例数量
        t_m_now = (m_value[i] + m_value[i + 1]) / 2 # 计算当前密度分割点
        t_t_now = (t_value[i] + t_value[i + 1]) / 2 # 计算当前含糖量分割点

        # 遍历valueset中的每个样本，根据分割点分配到相应的区间 
        for j in range(len(valueset)):
            print("Debug: valueset[j]:", valueset[j])
            if valueset[j][0] > t_m_now:
                count[0] += 1
                if valueset[j][0] == 1:  # # 假设'是'对应于1
                    true_count[0] += 1
            else:
                count[1] += 1
                if valueset[j][0] == 1: 
                    true_count[1] += 1
            if valueset[j][1] > t_t_now:
                count[2] += 1
                if valueset[j][1] == 1: 
                    true_count[2] += 1
            else:
                count[3] += 1
                if valueset[j][1] == 1:  
                    true_count[3] += 1
        # 计算当前分割点的熵值
        t_m_ent_now = base_entropy - (count[0] / len(valueset)) * calc_entropy(count[0], true_count[0]) - (count[1] / len(valueset)) * calc_entropy(count[1], true_count[1])
        t_t_ent_now = base_entropy - (count[2] / len(valueset)) * calc_entropy(count[2], true_count[2]) - (count[3] / len(valueset)) * calc_entropy(count[3], true_count[3])
        
        # 更新最佳分割点和对应的熵值
        if t_m_ent_now > t_ent[0]:
            t_ent[0] = t_m_ent_now
            t_m_div = t_m_now

        if t_t_ent_now > t_ent[1]:
            t_ent[1] = t_t_ent_now
            t_t_div = t_t_now

    # 根据最佳分割点更新valueset中的值，并更新labelset中的标签
    for value in valueset:
        value[0] = 0 if value[0] < t_m_div else 1
        value[1] = 0 if value[1] < t_t_div else 1

    for i in range(len(labelset)):
        labelset[i].insert(len(labelset[i]) - 1, valueset[i][0])  # 插入密度标签
        labelset[i].insert(len(labelset[i]) - 1, valueset[i][1])  # 插入含糖量标签

    # 创建标签列表，用于构建决策树
    t_label = ['含糖量<={0}'.format(t_t_div), '含糖量>{0}'.format(t_t_div)]
    m_label = ['密度<={0}'.format(t_m_div), '密度>{0}'.format(t_m_div)]
    total_labels.append(t_label)
    total_labels.append(m_label)

    return labelset

#计算labelset中每个特征的熵。找到使用该特征分割数据集时，信息增益最大的特征。
def get_label_entropy(labelset):
    base_entropy = calc_base_entropy(labelset) # 计算labelset的基础熵
    label_features = len(labelset[0]) - 1 # 特征数量，减1是因为最后一个元素是标签
    label_entropy = [] # 存储每个特征的熵值
    
    # 遍历labelset中的每个特征
    for i in range(label_features):
        featlist = [example[i] for example in labelset] # 获取当前特征的所有值
        unique_vals = list(set(featlist)) # 获取当前特征的唯一值
        new_entropy = base_entropy # 初始化新熵

        # 遍历当前特征的每个唯一值
        for val in unique_vals:
            count = [0] * len(unique_vals)  # 存储每个唯一值的样本数量
            true_count = [0] * len(unique_vals)  # 存储每个唯一值的正例数量

            # 根据当前特征的唯一值分割labelset
            for example in labelset:
                if example[i] == val:
                    index = unique_vals.index(val)
                    count[index] += 1
                    if example[-1] == '是':
                        true_count[index] += 1 # 假设标签'是'对应于1
           
            # 计算当前分割方式的熵值
            new_entropy -= sum((count[j] / len(labelset)) * calc_entropy(count[j], true_count[j]) for j in range(len(unique_vals)))

        label_entropy.append(new_entropy)

    return label_entropy

def divide_set(labelset, labels):
    div_sets = [] # 存储分割后的数据集子集
    label_entropy = get_label_entropy(labelset) # 计算labelset中每个特征的熵
    max_entropy = max(label_entropy) # 找到最大熵值
    max_index = label_entropy.index(max_entropy) # 找到最大熵值对应的特征索引
    best_feature = labels.pop(max_index)  # 获取最佳特征
    value = total_labels.pop(max_index) # 获取最佳特征的值
    featlist = [example[max_index] for example in labelset] # 获取最佳特征的值
    unique_vals = list(set(featlist))  # 获取当前特征的唯一值
    
    # 根据最佳特征的唯一值分割数据集
    for val in unique_vals:
        temp_set = [example for example in labelset if example[max_index] == val]
        for example in temp_set:
            labelset.remove(example)
            example.pop(max_index)
        div_sets.append(temp_set)
    
    return div_sets, best_feature, value

#从数据集中创建决策树的函数。递归地构建树，通过找到最佳特征来分割数据集。
def create_tree(labelset, labels, node_level):
    now_level = node_level + 1
    class_list = [example[-1] for example in labelset]
    
    # 如果数据集中所有样本的标签都相同，则返回该标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    div_sets, best_feature, values = divide_set(labelset, labels) # 分割数据集
    my_tree = {best_feature: {}} # 创建以最佳特征为根的决策树节点
    
    # 为分割后的每个子集创建子树
    for div_set in div_sets:
        value = values[div_sets.index(div_set)]
        my_tree[best_feature][value] = create_tree(div_set, labels, now_level)

    return my_tree

if __name__ == '__main__':
    labelset, valueset = get_data()
    labelset = value_to_label(labelset, valueset)
    label_entropy = get_label_entropy(labelset)
    my_tree = create_tree(labelset, labels, node_level)
    print(my_tree)