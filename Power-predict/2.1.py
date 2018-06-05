# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/5/30
   Description :    数据组合
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import itertools


X_num = pd.read_pickle('X_num.pkl')
X_num.head()


c = X_num.columns


# 数值型变量的加减乘除组合
Tab = list(itertools.combinations(range(0,len(X_num.columns)),2))

cols_add = [c[x] + '_add_' + c[y] for x,y in Tab]
cols_sub = [c[x] + '_sub_' + c[y] for x,y in Tab]
cols_mut = [c[x] + '_mut_' + c[y] for x,y in Tab]
cols_div = [c[x] + '_div_' + c[y] for x,y in Tab]
# 定义空的DataFrame
X_num_add = pd.DataFrame(np.zeros([len(X_num),len(Tab)]), index=X_num.index, columns=cols_add)
X_num_sub = pd.DataFrame(np.zeros([len(X_num),len(Tab)]), index=X_num.index, columns=cols_sub)
X_num_mut = pd.DataFrame(np.zeros([len(X_num),len(Tab)]), index=X_num.index, columns=cols_mut)
X_num_div = pd.DataFrame(np.zeros([len(X_num),len(Tab)]), index=X_num.index, columns=cols_div)

for i,j in Tab:
    X_num_add[c[i] + '_add_' + c[j]] = X_num[i] + X_num[j]
    X_num_sub[c[i] + '_sub_' + c[j]] = X_num[i] - X_num[j]
    X_num_mut[c[i] + '_mut_' + c[j]] = X_num[i] * X_num[j]
    X_num_div[c[i] + '_div_' + c[j]] = X_num[i] / X_num[j]

# 保存组合变量
X_num_add.to_pickle('X_num_add.pkl')
X_num_sub.to_pickle('X_num_sub.pkl')
X_num_mut.to_pickle('X_num_mut.pkl')
X_num_div.to_pickle('X_num_div.pkl')


# 数值型变量增加移动平均/移动标准差
windows = [5, 10, 20, 30]

for i,window in enumerate(windows):

    X_num_mov_mean = X_num.rolling(window, min_periods=1).mean()
    X_num_mov_mean.columns = [x + 'mean' + str(window) for x in X_num.columns]
    X_num_mov_std = X_num.rolling(window, min_periods=1).std()
    X_num_mov_std.columns = [x + 'std' + str(window) for x in X_num.columns]

    if i == 0:
        X_num_move = pd.concat([X_num_mov_mean, X_num_mov_std], axis=1)
    else:
        X_num_move1 = pd.concat([X_num_mov_mean, X_num_mov_std], axis=1)
        X_num_move = pd.concat([X_num_mov, X_num_move1], axis=1)

# 保存move文件
X_num_mov.to_pickle('X_num_mov.pkl')






















