# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/8
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
    os.chdir(r'D:\00_工作日志\O\2018-06\赤峰项目故障\功率相关\feature')
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

def model_predict(X_train, y_train, X_test, y_test):
    "模型训练及预测"
    model = lgb.LGBMRegressor(colsample_bytree=0.607,
                              learning_rate=0.1,
                              n_estimators=150,
                              n_jobs=-1,
                              num_leaves=90,
                              reg_alpha=0.556,
                              reg_lambda=0.778,
                              subsample=0.82,
                              subsample_freq=2)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print('train r2: ', r2_score(y_train, y_train_pred))
    print('test r2: ', r2_score(y_test, y_test_pred))
    return y_train_pred, y_test_pred

def error_df_set(df_real, pred):
    "建立误差df，并增加季节、月份、天的类别，便于groupby"
    df_error = pd.DataFrame(data=df_real.values-pred, index=df_real.index, columns=['error'])
    df_error['real'] = df_real.values
    df_error['pred'] = pred
    #
    df_error['year_season'] = [str(x)[: 4] + '_' + str(math.floor((int(str(x)[5: 7])-0.1)/3+1)) for x in df_error.index]
    df_error['year_month'] = [str(x)[: 7] for x in df_error.index]
    df_error['year_day'] = [str(x)[: 10] for x in df_error.index]
    return df_error

def error_timing(df_error, ds, dt, name='name'):
    "误差时序图"
    c = df_error.copy() # 创建副本
    c.index = pd.to_datetime(c.index)
    c = c.loc[ds:dt,:]
    plt.figure(figsize=(16,9))
    plt.subplot(2, 1, 1)
    plt.plot(c.index, c['real'], alpha=0.8, label='real power')
    plt.plot(c.index, c['pred'], alpha=0.8, label='pred power')
    plt.subplot(2, 1, 2)
    plt.plot(c.index, c['real']-c['pred'], label='error')
    plt.legend()
    plt.savefig(name + '_figure1_误差时序图.png',dpi=300)

def error_mean_std_fig(df_error, t='year_month', name='name'):
    "绘制分仓mean与std图"
    gt_mean = df_error['error'].groupby(df_error[t]).mean()
    gt_std = df_error['error'].groupby(df_error[t]).std()
    plt.figure(figsize=(16,9))
    plt.subplot(211)
    plt.bar(gt_mean.index, gt_mean.values, label='mean')
    plt.subplot(212)
    plt.bar(gt_std.index, gt_std.values, label='std')
    plt.savefig(name + '_figure2_分仓mean与std图.png',dpi=300)

def error_violin_fig(df_error, t='year_month', name='name'):
    "绘制violin图与箱线图"

    # 将Series整理成numpy列表
    all_data = []
    for i in sorted(list(set(df_error[t]))):
        b = df_error[df_error[t]==i]
        b = b['error']
        all_data.append(b.values)

    # 画图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))

    # plot violin plot
    axes[0].violinplot(all_data,
                       showmeans=False,
                       showmedians=True)
    axes[0].set_title('Violin plot')

    # plot box plot
    axes[1].boxplot(all_data)
    axes[1].set_title('Box plot')

    # adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(all_data))])
        ax.set_ylabel('error')

    # add x-tick labels
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=sorted(list(set(df_error[t]))))
    plt.savefig(name + '_figure3_violin图与箱线图.png',dpi=300)

def real_pred_fig(df_error, X_test, base='r72'):
    "绘制真实(x)与预测(y)的功率曲线，基于变桨角度、风速等着色"
    c = df_error.copy()
    try:
        c[base] = X_test[base].values
        c.plot(kind='scatter', x='real', y='pred', alpha=0.4,
               c=base, cmap=plt.get_cmap('jet'), colorbar=True)
        plt.show()
    except:
        c.plot(kind='scatter', x='real', y='pred', alpha=0.4)
        plt.show()



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

    # 模型预测
    y_train_pred, y_test_pred = model_predict(X_train, y_train, X_test, y_test)

    os.chdir(r'D:\00_工作日志\O\2018-06\赤峰项目故障\功率相关\01')

    # 作图展示训练集
    df_error_train = error_df_set(y_train, y_train_pred)
    error_timing(df_error_train, '2016-01-01', '2016-12-31', name='train')
    error_mean_std_fig(df_error_train, t='year_month', name='train')
    error_violin_fig(df_error_train, t='year_month', name='train')
    # real_pred_fig(df_error_train, X_train, base='r76')

    # 作图展示测试集
    df_error_test = error_df_set(y_test, y_test_pred)
    error_timing(df_error_test, '2017-01-01', '2017-12-31', name='test')
    error_mean_std_fig(df_error_test, t='year_day', name='test')
    error_violin_fig(df_error_test, t='year_day', name='test')
    # real_pred_fig(df_error_test, X_test, base='r76')




