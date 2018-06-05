# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/5/30
   Description :    数据转换
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import itertools


def f_num_dif(X_num):
    "离散差值"
    X_num_diff = pd.DataFrame(data=np.diff(X_num, axis=0), index=X_num.index[1:], columns=X_num.columns + '_diff')
    return X_num_diff


def f_num_abs(X_num):
    "取绝对值"
    X_num_abs = np.abs(X_num)
    X_num_abs.columns = X_num.columns + '_abs'
    return X_num_abs


def f_num_log(X_num_abs):
    "取log"
    X_num_log = np.log(X_num_abs + 1)
    X_num_log.columns = X_num.columns + '_log'
    return X_num_log


def f_num_rec(X_num):
    "倒数"
    X_num_rec = 1 / X_num
    X_num_rec.columns = X_num.columns + '_rec'
    return X_num_rec


def f_num_sqr(X_num_abs):
    "开方"
    X_num_sqr = np.power(X_num_abs, 0.5)
    X_num_sqr.columns = X_num.columns + '_sqr'
    return X_num_sqr


if __name__ == '__main__':

    # 读取数据
    X_num = pd.read_pickle('X_num.pkl')
    #X_num.head()

    # 保存离散插值
    X_num_diff = f_num_abs(X_num)
    X_num_diff.to_pickle('X_num_diff.pkl')

    # 保存绝对值
    X_num_abs = f_num_abs(X_num)
    X_num_abs.to_pickle('X_num_abs.pkl')

    # 保存log值
    X_num_log = f_num_abs(X_num_abs)
    X_num_log.to_pickle('X_num_log.pkl')

    # 保存倒数
    X_num_rec = f_num_rec(X_num)
    X_num_rec.to_pickle('X_num_rec.pkl')

    # 保存开方
    X_num_sqr = f_num_rec(X_num)
    X_num_sqr.to_pickle('X_num_sqr.pkl')


