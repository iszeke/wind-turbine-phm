# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/5/30
   Description :    特征重要性(transform_move)
-------------------------------------------------
"""
import pandas as pd
import numpy as np
import os
import xgboost
from sklearn.metrics import mean_squared_error

def feature_choice(X_train, y_train, X_test, y_test, save_name):
    "依据集成方法对特征进行选择"

    train = xgboost.DMatrix(data=X_train, label=y_train)
    test = xgboost.DMatrix(data=X_test, label=y_test)

    params = {'max_depth': 3,
              'learning_rate': 0.8,
              'objective': 'reg:linear',
              'silent': False,
              'gamma': 10,
              'min_child_weight': 5,
              'max_delta_step': 0,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'reg_alpha': 0,
              'reg_lambda': 0,
              'scale_pos_weight': 1,
              'base_score': 0.5,
              'seed': 123}

    model = xgboost.train(params,train,num_boost_round=100,verbose_eval=True)

    y_train_pred = model.predict(train)
    y_test_pred = model.predict(test)

    print('MSE_train: %0.3f' % mean_squared_error(y_train, y_train_pred))
    print('MSE_test: %0.3f' % mean_squared_error(y_test, y_test_pred))

    # 显示特征的重要性
    xgboost.plot_importance(model)
    print(model.get_fscore())
    # 保存特征重要性表
    df_imp = pd.DataFrame({'feature_item': list(model.get_fscore().keys()),
                           'importance': list(model.get_fscore().values())})

    df_imp = df_imp.sort_values('importance', ascending=False)
    df_imp.to_csv(save_name, header=True, encoding='ASCII', index=False)


if __name__ == '__main__':
    # 读取数据
    os.chdir(r'')
    y = pd.read_pickle('y.pkl')

    X_cat = pd.read_pickle('X_cat.pkl')
    X_num = pd.read_pickle('X_num.pkl')

    X_num_add = pd.read_pickle('X_num_add.pkl')
    X_num_sub = pd.read_pickle('X_num_sub.pkl')
    X_num_mut = pd.read_pickle('X_num_mut.pkl')
    X_num_div = pd.read_pickle('X_num_div.pkl')

    df = pd.concat([y, X_num, X_num_add, X_num_sub, X_num_mut, X_num_div], axis=1)

    df = df.dropna(how='all', axis=1)  # 删除所有为0的列

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

    # 划分训练集与测试集
    df_train = df['2016-01-01':'2016-12-31']
    df_test = df['2017-01-01':'2017-12-31']

    X_train = np.array(df_train.ix[:, 1:])
    y_train = np.array(df_train.ix[:, 0])
    X_test = np.array(df_test.ix[:, 1:])
    y_test = np.array(df_test.ix[:, 0])

    # 调用函数得到特征重要性排名
    # 调用函数得到特征重要性排名
    save_name = 'df_imp_transform_move.csv'
    feature_choice(X_train, y_train, X_test, y_test, save_name)











































