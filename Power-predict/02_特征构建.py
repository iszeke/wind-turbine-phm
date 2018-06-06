# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/6
   Description :    1/ 暴力数据组合(加、减、乘、除)
                    2/ 暴力移动平均，移动标准差, 窗口[5, 10, 20, 30]
                    3/ 暴力数据转换(相邻差, 绝对值, 平方, 开方, 倒数, Log等)
-------------------------------------------------
"""

import pandas as pd
import numpy as np
import itertools
import os

class FeatureConstruct(object):

    def __init__(self, X_num):
        self.X_num = X_num

    def combine(self):
        "特征之间的加减乘除组合"

        c = self.X_num.columns
        Tab = list(itertools.combinations(range(0, len(X_num.columns)), 2))

        # 定义列名
        cols_add = [c[x] + '_add_' + c[y] for x, y in Tab]
        cols_sub = [c[x] + '_sub_' + c[y] for x, y in Tab]
        cols_mut = [c[x] + '_mut_' + c[y] for x, y in Tab]
        cols_div = [c[x] + '_div_' + c[y] for x, y in Tab]

        # 定义空的DataFrame
        X_num_add = pd.DataFrame(np.zeros([len(self.X_num), len(Tab)]), index=self.X_num.index, columns=cols_add)
        X_num_sub = pd.DataFrame(np.zeros([len(self.X_num), len(Tab)]), index=self.X_num.index, columns=cols_sub)
        X_num_mut = pd.DataFrame(np.zeros([len(self.X_num), len(Tab)]), index=self.X_num.index, columns=cols_mut)
        X_num_div = pd.DataFrame(np.zeros([len(self.X_num), len(Tab)]), index=self.X_num.index, columns=cols_div)

        # 组合数据
        for i, j in Tab:
            X_num_add[c[i] + '_add_' + c[j]] = self.X_num.iloc[:, i] + self.X_num.iloc[:, j]
            X_num_sub[c[i] + '_sub_' + c[j]] = self.X_num.iloc[:, i] - self.X_num.iloc[:, j]
            X_num_mut[c[i] + '_mut_' + c[j]] = self.X_num.iloc[:, i] * self.X_num.iloc[:, j]
            X_num_div[c[i] + '_div_' + c[j]] = self.X_num.iloc[:, i] / (self.X_num.iloc[:, j]+0.01)

        # 合并所有
        X_num_combine = pd.concat([X_num_add, X_num_sub, X_num_mut, X_num_div], axis=1)
        X_num_combine.to_pickle('X_num_combine.pkl')
        print('\ncombine --- complete !')
        print('combine shape: ', X_num_combine.shape)

    def move(self):
        "特征之间的移动平均/移动标准差"

        windows = [5, 10, 20, 30]
        for i, window in enumerate(windows):
            X_num_mov_mean = self.X_num.rolling(window, min_periods=1).mean()
            X_num_mov_mean.columns = [x + '_mean_' + str(window) for x in self.X_num.columns]
            X_num_mov_std = self.X_num.rolling(window, min_periods=1).std()
            X_num_mov_std.columns = [x + '_std_' + str(window) for x in self.X_num.columns]
            if i == 0:
                X_num_move = pd.concat([X_num_mov_mean, X_num_mov_std], axis=1)
            else:
                X_num_move1 = pd.concat([X_num_mov_mean, X_num_mov_std], axis=1)
                X_num_move = pd.concat([X_num_move, X_num_move1], axis=1)
        # 保存move文件
        X_num_move.to_pickle('X_num_move.pkl')
        print('\nmove --- complete !')
        print('move shape: ', X_num_move.shape)

    def transform(self):
        "对数据做一些变换"

        # 相邻差值
        X_num_diff = pd.DataFrame(data=np.diff(self.X_num.values, axis=0), index=self.X_num.index[1:], columns=self.X_num.columns + '_diff')
        # 绝对值
        X_num_abs = pd.DataFrame(data=np.abs(self.X_num.values), index=self.X_num.index, columns=self.X_num.columns + '_abs')
        # 取log
        X_num_log = pd.DataFrame(data=np.log(np.abs(self.X_num.values)+0.01), index=self.X_num.index, columns=self.X_num.columns + '_log')
        # 取倒数
        X_num_rec = pd.DataFrame(data=1/(self.X_num.values+0.01), index=self.X_num.index, columns=self.X_num.columns + '_rec')
        # 取开方
        X_num_sqr = pd.DataFrame(data=np.power(np.abs(self.X_num.values), 0.5), index=self.X_num.index, columns=self.X_num.columns + '_sqr')
        # 取平方
        X_num_squ = pd.DataFrame(data=np.power(self.X_num.values, 2), index=self.X_num.index, columns=self.X_num.columns + '_squ')

        # 合并所有
        X_num_transform = pd.concat([X_num_diff, X_num_abs, X_num_log, X_num_rec, X_num_sqr, X_num_squ], axis=1)
        # 保存文件
        X_num_transform.to_pickle('X_num_transform.pkl')
        print('\ntransform --- complete !')
        print('transform shape: ', X_num_transform.shape)


if __name__ == '__main__':

    os.chdir(r'C:\Users\Chinawindey\Desktop\123')
    X_num = pd.read_pickle('X_num.pkl')
    print(X_num.shape)

    fc = FeatureConstruct(X_num)
    fc.combine()
    fc.move()
    fc.transform()
    print('FeatureConstruct Complete !')

