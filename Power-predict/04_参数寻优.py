# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/7
   Description :    调参
-------------------------------------------------
"""

import os
import pandas as pd
from hyperopt import fmin, tpe, hp
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score



def combine_feature():
    "将特征筛选后的特征进行组合"
    df_inf_combine = pd.read_csv('importance_of_combine.csv')
    df_inf_move = pd.read_csv('importance_of_move.csv')
    df_inf_transform = pd.read_csv('importance_of_transform.csv')

    df_inf = pd.concat([df_inf_combine, df_inf_move, df_inf_transform], axis=0)
    print(df_inf.shape)
    df_fn = df_inf['feature_name']
    df_fn = df_fn.drop_duplicates() # 删除重复的行
    print(df_fn.shape)
    return df_fn.values.tolist()


def train_and_test(df, X_cat):
    # 筛选风机状态为3
    df = df[X_cat['r3'] == 3]

    # 如果有inf，则置为nan
    df[df == np.inf] = np.nan

    # 删除所有为nan的列
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='any', axis=0)

    # 筛选掉全为相同值的列
    df = df.loc[:, df.apply(lambda x: len(np.unique(x)), axis=0) != 1]

    # 划分训练集与测试集
    df.index = pd.to_datetime(df.index)
    df_train = df.loc['2016-01-01':'2016-12-31', :]
    df_test = df.loc['2017-01-01':'2017-12-31', :]

    X_train = df_train.iloc[:, 1:]
    y_train = df_train.iloc[:, 0]
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]

    return X_train, y_train, X_test, y_test



if __name__ == '__main__':

    os.chdir(r'D:\00_工作日志\O\2018-06\赤峰项目故障\功率相关\01')

    y = pd.read_pickle('y.pkl')
    X_cat = pd.read_pickle('X_cat.pkl')
    X_num = pd.read_pickle('X_num.pkl')
    X_dum = pd.read_pickle('X_dum.pkl')
    X_num_combine = pd.read_pickle('X_num_combine.pkl')
    X_num_move = pd.read_pickle('X_num_move.pkl')
    X_num_transform = pd.read_pickle('X_num_transform.pkl')

    df_all = pd.concat([y, X_num, X_dum, X_num_combine, X_num_move, X_num_transform], axis=1)

    # 使用特征筛选后的重要特征
    df_all = df_all.loc[:, ['r21']+combine_feature()]
    # 训练集与测试集分割
    X_train, y_train, X_test, y_test = train_and_test(df_all, X_cat)




    model = lgb.LGBMRegressor(seed=123)


    param_dist = {'num_leaves': range(20, 100, 5),
                  #'learning_rate': np.linspace(0.01,2,20),
                  'n_estimators': range(100,1000,50),
                  'colsample_bytree': np.linspace(0.5,0.98,10),
                  'subsample': np.linspace(0.5,0.98,10),
                  'subsample_freq': range(1, 4),
                  'reg_alpha': np.linspace(0, 1, 10),
                  'reg_lambda': np.linspace(0, 1, 10),
                  }

    grid = RandomizedSearchCV(model, param_dist, cv=4, scoring='neg_mean_squared_error',n_iter=30, n_jobs=-1)

    #在训练集上训练
    grid.fit(X_train,y_train)
    #返回最优的训练器
    best_estimator = grid.best_estimator_
    print(best_estimator)
    #输出最优训练器的精度
    print(grid.best_score_)

    # 预测结果
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)
    print('\n')

    print('MSE_train: %0.3f' % r2_score(y_train, y_train_pred))
    print('MSE_test: %0.3f' % r2_score(y_test, y_test_pred))

    import matplotlib.pyplot as plt

    plt.subplot(2, 1, 1)
    plt.plot(X_train.index, y_train - y_train_pred)
    plt.subplot(2, 1, 2)
    plt.plot(X_test.index, y_test - y_test_pred)
    plt.show()

    # 训练集作图
    plt.subplot(2, 1, 1)
    plt.plot(X_train.index, y_train, label='real power')
    plt.plot(X_train.index, y_train_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(X_train.index, y_train - y_train_pred, label='error')
    plt.legend()
    plt.show()

    # 预测并作图
    plt.subplot(2, 1, 1)
    plt.plot(X_test.index, y_test, label='real power')
    plt.plot(X_test.index, y_test_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(X_test.index, y_test - y_test_pred, label='error')
    plt.legend()
    plt.show()















































