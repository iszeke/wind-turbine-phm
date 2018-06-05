# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/5/30
   Description :    特征重要性(所有特征)
-------------------------------------------------
"""
import pandas as pd
import numpy as np
import os
import xgboost
from sklearn.metrics import mean_squared_error


df_imp_sip = pd.read_csv('df_imp_transform_move.csv')
df_imp_mul = pd.read_csv('df_imp_interaction.csv')
col = list(np.union1d(list(df_imp_sip.ix[:,'feature_name']),list(df_imp_mul.ix[:,'feature_name'])))

y = pd.read_pickle('y.pkl')

X_ord = pd.read_pickle('X_ord.pkl')
X_cat = pd.read_pickle('X_cat.pkl')
X_dum = pd.read_pickle('X_dum.pkl')

X_num = pd.read_pickle('X_num.pkl')

X_num_dif = pd.read_pickle('X_num_dif.pkl')
X_num_abs = pd.read_pickle('X_num_abs.pkl')
X_num_log = pd.read_pickle('X_num_log.pkl')
X_num_rec = pd.read_pickle('X_num_rec.pkl')
X_num_sqr = pd.read_pickle('X_num_sqr.pkl')
X_num_squ = pd.read_pickle('X_num_squ.pkl')

X_num_add = pd.read_pickle('X_num_add.pkl')
X_num_sub = pd.read_pickle('X_num_sub.pkl')
X_num_mut = pd.read_pickle('X_num_mut.pkl')
X_num_div = pd.read_pickle('X_num_div.pkl')


df = pd.concat([y,
                X_num,
                X_ord,
                X_cat,
                X_dum,
                X_num_dif,
                X_num_abs,
                X_num_log,
                X_num_rec,
                X_num_sqr,
                X_num_squ,
                X_num_add,
                X_num_mut,
                X_num_div],axis=1)

df = df.dropna(how='all', axis=1)  # 删除所有为0的列

df = df.loc[:,col]

# 筛选风机状态
cond = (X_cat.loc[:, 'r3'] == 6)
df = df[cond == 1, :]

# 如果有无限值，则置为nan
for k in range(len(df.columns)):
    pos = np.isinf(df.ix[:, k])
    df.ix[pos, k] = np.nan

df = df.dropna(how='any', axis=0)

df.index = pd.to_datetime(df.index)
df = df.loc[:, df.apply(lambda x: len(np.unique(x)), axis=0) != 1]

df.to_pickle('df_comb.pkl')























