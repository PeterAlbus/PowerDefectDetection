import numpy as np
import pandas as pd

data = pd.read_csv('result/final/2.csv', index_col=0)


def minmax_positive(x):
    # x = np.array(list(map(float, x)))
    return (x - x.min()) / (x.max() - x.min())


def minmax_negative(x):
    # x = np.array(list(map(float, x)))
    return (x.max() - x) / (x.max() - x.min())


positive_rows = ['acc/ca', 'ACTC', 'BD', 'ALDp1', 'ALDp2', 'ALDp3']
negative_rows = ['ACAC', 'NTE',  'ENI']
data = data.drop('ALDp',axis=1)
data = data.drop('ACC',axis=1)
data = data.drop('CA',axis=1)
data = data.apply(lambda x: minmax_positive(x) if x.name in positive_rows else x)
data = data.apply(lambda x: minmax_negative(x) if x.name in negative_rows else x)
# 保存数据到csv
data.to_csv('result/final/3.csv')

p_ij = data/data.sum()
print(p_ij.head())

p_ij = p_ij.replace(0, 1e-100)
ei = -1 / np.log(len(data)) * np.sum(p_ij * np.log(p_ij), axis = 0)
print(ei)

di = 1 - ei
print(di)

wi = di / np.sum(di)
print(wi)
