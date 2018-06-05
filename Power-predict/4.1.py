# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/5
   Description :    模型1
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.model_selection import cross_val_score

def model(j):
    "模型"
    if j == 1:
        model = ExtraTreesRegressor(n_estimators=200,
                                    max_depth=3,
                                    n_jobs=-1,
                                    random_state=124)
    if j == 2:
        model = RandomForestRegressor(n_estimators=200,
                                      max_depth=3,
                                      n_jobs=-1,
                                      random_state=124)
    if j == 3:
        model = xgboost.XGBRegressor(max_depth=3,
                                     learning_rate=0.8,
                                     n_estimators=200,
                                     silent=False,
                                     objective='reg:linear',
                                     nthread=-1,
                                     gamma=10,
                                     min_child_weight=5,
                                     max_delta_step=0,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     reg_alpha=0,
                                     reg_lambda=0,
                                     seed=123,
                                     missing=None)
    return model


if __name__ == '__main__':

    os.chdir(r'')
    df = pd.read_pickle('df_comb.pkl')

    # 划分训练集与测试集
    df_train = df['2016-01-01':'2016-12-31']
    df_test = df['2017-01-01':'2017-12-31']

    X_train = np.array(df_train.ix[:, 1:])
    y_train = np.array(df_train.ix[:, 0])
    X_test = np.array(df_test.ix[:, 1:])
    y_test = np.array(df_test.ix[:, 0])

    for j in [1, 2, 3]:
        df_eval = cross_val_score(model(j), X_train, y_train, scoring='neg_mean_squared_erro', cv=5)
        print(df_eval)

































