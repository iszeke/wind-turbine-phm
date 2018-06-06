# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/6
   Description :    使用XGBoost筛选重要的特征
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


def feature_choice(X_train, y_train, X_test, y_test, save_name):
    "依据集成方法对特征进行选择"

    train = xgboost.DMatrix(data=X_train.values, label=y_train.values)
    test = xgboost.DMatrix(data=X_test.values, label=y_test.values)

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
    # 预测结果
    y_train_pred = model.predict(train)
    y_test_pred = model.predict(test)
    print('\n')
    print(save_name)
    print('MSE_train: %0.3f' % r2_score(y_train, y_train_pred))
    print('MSE_test: %0.3f' % r2_score(y_test, y_test_pred))

    # 显示特征的重要性
    xgboost.plot_importance(model)
    print(model.get_fscore())
    # 保存特征重要性表
    df_imp = pd.DataFrame({'feature_item': list(model.get_fscore().keys()),
                           'importance': list(model.get_fscore().values())})

    df_imp = df_imp.sort_values('importance', ascending=False)

    df_imp['feature_item'] = df_imp['feature_item'].apply(lambda x: int(x[1:]))

    df_imp['feature_name'] = X_train.columns[df_imp['feature_item']]

    df_imp.to_csv(save_name + '.csv', header=True, encoding='ASCII', index=False)



if __name__ == '__main__':

    # 读取数据
    os.chdir(r'C:\Users\Chinawindey\Desktop\123')
    y = pd.read_pickle('y.pkl')
    X_cat = pd.read_pickle('X_cat.pkl')
    X_num = pd.read_pickle('X_num.pkl')
    X_dum = pd.read_pickle('X_dum.pkl')
    X_num_combine = pd.read_pickle('X_num_combine.pkl')
    X_num_move = pd.read_pickle('X_num_move.pkl')
    X_num_transform = pd.read_pickle('X_num_transform.pkl')


    # 使用那些特征
    df_combine = pd.concat([y, X_num, X_dum, X_num_combine], axis=1)
    df_move = pd.concat([y, X_num, X_dum, X_num_move], axis=1)
    df_transform = pd.concat([y, X_num, X_dum, X_num_transform], axis=1)
    df_all = pd.concat([y, X_num, X_dum, X_num_combine, X_num_move, X_num_transform], axis=1)


    # 仅使用combine
    a = train_and_test(df_combine, X_cat)
    feature_choice(a[0], a[1], a[2], a[3], 'importance_of_combine')

    # 仅使用move
    a = train_and_test(df_move, X_cat)
    feature_choice(a[0], a[1], a[2], a[3], 'importance_of_move')

    # 仅使用transform
    a = train_and_test(df_transform, X_cat)
    feature_choice(a[0], a[1], a[2], a[3], 'importance_of_transform')

    # 全部使用
    a = train_and_test(df_all, X_cat)
    feature_choice(a[0], a[1], a[2], a[3], 'importance_of_all')












































