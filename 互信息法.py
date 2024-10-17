from numpy import set_printoptions
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy as np

data = pd.read_excel
X = data.drop(['',],axis=1)
Y = data['']
np.set_printoptions(suppress=True) #直接显示数字而不是科学计数法
mutual_info_classif(X, Y)

test = SelectKBest(score_func=mutual_info_classif, k=3)
fit = test.fit(X,Y)
X.columns
features = fit.transform(X)