# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/7
   Description :    预测展示
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import math
from datetime import datetime
from sklearn.metrics import mean_squared_error

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


def bins_error_fig(df_error):

    # 季节平均误差
    df_error['year_season'] = [str(x)[: 4] + '_' + str(math.floor((int(str(x)[5: 7])-0.1)/3+1)) for x in df_error.index]
    g_season_mean = df_error['error'].groupby(df_error['year_season']).mean()
    g_season_std = df_error['error'].groupby(df_error['year_season']).std()
    plt.subplot(211)
    plt.bar(g_season_mean.index, g_season_mean.values, label='mean')
    plt.subplot(212)
    plt.bar(g_season_std.index, g_season_std.values, label='std')
    plt.show()

    # 月份平均误差
    df_error['year_month'] = [str(x)[: 7] for x in df_error.index]
    g_month_mean = df_error['error'].groupby(df_error['year_month']).mean()
    g_month_std = df_error['error'].groupby(df_error['year_month']).std()
    plt.subplot(211)
    plt.bar(g_month_mean.index, g_month_mean.values, label='mean')
    plt.subplot(212)
    plt.bar(g_month_std.index, g_month_std.values, label='std')
    plt.show()


    # # violin
    a = sorted(list(set(df_error['year_month'])))
    c = []
    for i in a:
        b = df_error[df_error['year_month']==i]
        b = b['error']
        c.append(b.values)
    print(len(c))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 10))

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # plot violin plot
    axes[0].violinplot(c,
                       showmeans=False,
                       showmedians=True)
    axes[0].set_title('Violin plot')

    # plot box plot
    axes[1].boxplot(c)
    axes[1].set_title('Box plot')

    # adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(c))])
        ax.set_xlabel('Four separate samples')
        ax.set_ylabel('Observed values')

    # add x-tick labels
    plt.setp(axes, xticks=[y + 1 for y in range(len(c))],
             xticklabels=a)
    plt.show()








    # 天平均误差
    df_error['year_day'] = [str(x)[: 10] for x in df_error.index]
    g_day_mean = df_error['error'].groupby(df_error['year_day']).mean()
    g_day_std = df_error['error'].groupby(df_error['year_day']).std()
    plt.subplot(211)
    plt.bar(g_day_mean.index, g_day_mean.values, label='mean')
    plt.subplot(212)
    plt.bar(g_day_std.index, g_day_std.values, label='std')
    plt.show()









if __name__ == '__main__':


    os.chdir(r'C:\Users\Chinawindey\Desktop\123')

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


    model = lgb.LGBMRegressor(colsample_bytree=0.927,
                              learning_rate=0.1,
                              n_estimators=250,
                              n_jobs=-1,
                              num_leaves=25,
                              reg_alpha=0.778,
                              reg_lambda=0.111,
                              subsample=0.98,
                              subsample_freq=2)

    model.fit(X_train, y_train)


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(r2_score(y_train, y_train_pred))
    print(r2_score(y_test, y_test_pred))



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
    plt.subplot(3, 1, 1)
    plt.plot(X_test.index, y_test, label='real power')
    plt.plot(X_test.index, y_test_pred, label='predict power')
    plt.subplot(3, 1, 2)
    plt.plot(X_test.index, y_test - y_test_pred, label='error')
    plt.subplot(3, 1, 3)
    df_error = pd.DataFrame(data = y_test.values-y_test_pred, index=X_test.index, columns=['error'])
    df_error_mean = df_error.rolling(10, min_periods=1).mean()
    plt.plot(df_error_mean, label='mean_error')
    plt.legend()
    plt.show()

    plt.scatter(y_train, y_train_pred, label='real power')
    plt.show()

    bins_error_fig(df_error)









