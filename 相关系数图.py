import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
filepath = '01047000.csv'
data = pd.read_csv(filepath)

# 解析第一列为datetime格式
data['timestamp'] = pd.to_datetime(data.iloc[:, 0])  # 第一列为时间列

# 将时间转换为时间戳（数值格式）
data['timestamp'] = data['timestamp'].map(pd.Timestamp.timestamp)

# 计算包含时间戳的皮尔逊相关系数
cor_time = data.corr(method='pearson')  # 计算相关系数
print(cor_time)  # 输出相关系数

# 设置字体和绘图参数
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(font_scale=0.7, rc=rc)  # 设置字体大小

# 绘制包含时间相关性的热力图
sns.heatmap(cor_time,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            linewidth=0.5,  # 设置每个单元格的距离
            linecolor='blue',  # 设置间距线的颜色
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            cmap='coolwarm_r',  # 设置热力图颜色
            )

# 保存和显示图片
plt.title("01047000.csv")
plt.savefig("Pearson_correlation_01047000.png", dpi=800)
plt.ion()  # 显示图片
