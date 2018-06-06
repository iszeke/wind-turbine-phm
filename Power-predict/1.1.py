# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/6/6
   Description :    数据基本整理及保存
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import os
import math

def data_read(file_name, cols):
    "读取文件到DataFrame"
    df = pd.read_csv(file_name, low_memory=False)
    df = df.loc[:, cols]    # 切片选择的列
    # index改为时间
    df.index = df.loc[:, 'r1']
    df.index = pd.to_datetime(df.index)
    return df

def base_filter(df):
    "数据的基本整理"

    # 1、取值范围筛选
    df[(df['r21'] < 5) & (df['r21'] > 2200)] = np.nan # 功率异常值
    df[(df['r43'] < 0.05) & (df['r43'] > 50)] = np.nan # 筛选掉风速大于50
    df[(df['r46'] < 0) & (df['r46'] > 360)] = np.nan  # 筛选掉风向在360°以外
    df[(df['r86'] < -50) & (df['r86'] > 50)] = np.nan # 筛选掉机舱外温度在50度以上
    df[(df['r60'] < -180) & (df['r60'] > 180)] = np.nan  # 筛选掉风向在360°以外
    df[(df['r72'] < -5) & (df['r72'] > 92)] = np.nan  # 筛选桨距角1
    df[(df['r73'] < -5) & (df['r73'] > 92)] = np.nan  # 筛选桨距角2
    df[(df['r74'] < -5) & (df['r74'] > 92)] = np.nan  # 筛选桨距角3
    df[(df['r76'] < -5) & (df['r76'] > 92)] = np.nan  # 筛选桨距角3

    # 2、删除所有带空值的行
    df = df.dropna(how='any', axis=0)

    return df

def radial_transform(df):
    "度数值转化为弧度值"
    # 风向，机舱位置增加sin与cos值
    df['r46_cos'] = df.loc[:, 'r46'].apply(lambda x: math.cos(math.radians(x)))
    df['r46_sin'] = df.loc[:, 'r46'].apply(lambda x: math.sin(math.radians(x)))

    df['r61_cos'] = df.loc[:, 'r61'].apply(lambda x: math.cos(math.radians(x)))
    df['r61_sin'] = df.loc[:, 'r61'].apply(lambda x: math.sin(math.radians(x)))

    df.drop(['r46'], axis=1, inplace=True) # 删除风向原始量
    df.drop(['r61'], axis=1, inplace=True) # 删除机舱位置原始量

    return df

def temp_transform(df):
    "将摄氏温度转换为绝对温度"
    df['r86'] = df['r86'] + 273.15
    df['r76'] = df['r76'] + 273.15
    df['r78'] = df['r78'] + 273.15
    df['r80'] = df['r80'] + 273.15
    return df

def yaw_error_abs(df):
    "将偏航误差转换成绝对值"
    df['r60'] = np.abs(df['r60'])

def numerical_X_save(df):
    "保存数值型特征"
    col_num = ['r43', 'r44', 'r45',  # 风速，平均风速30s，平均风速10min
               'r46_cos', 'r46_sin',  # 平均风向30s
               'r61_cos', 'r46_sin',  # 机舱位置
               'r86',  # 环境温度
               'r60',  # 偏航夹角
               'r72', 'r73', 'r74',  # 3个叶片桨距角实际值
               'r76', 'r78', 'r80',  # 3个变桨电机温度值'
               ]
    df.loc[:, col_num].to_pickle('X_num.pkl')

def category_X_save(df):
    "增加季节与月份哑向量"
    df['month'] = df.loc[:, 'r1'].apply(lambda x: int(x[5:7]))
    df['season'] = (df.loc[:, 'month'] / 4 + 1).apply(math.floor)
    df1 = pd.get_dummies(df['month'], prefix='month')
    df2 = pd.get_dummies(df['season'], prefix='season')
    df_cat = pd.concat([df1, df2], axis=1)

    # 保存类别特征
    df_cat.to_pickle('X_dum.pkl')

def state_X_save(df):
    "保存状态特征"
    df.loc[:, ['r3', 'r59']].to_pickle('X_cat.pkl')

def Y_save(df):
    "保存标签y"
    df.loc[:, 'r21'].to_pickle('y.pkl')


if __name__ == '__main__':

    # 定义文件读取路径
    os.chdir(r'D:\00_工作日志\O\2018-06\赤峰项目故障\data\赤峰二期5min数据')
    file_name = 'CFEQ_wt1_5min_201601-201712.csv'

    # 定义文件保存路径
    save_path = r'C:\Users\Chinawindey\Desktop\新建文件夹 (2)'

    # 筛选的变量
    cols = ['r1', # 时间
            'r43','r44','r45', # 风速，平均风速30s，平均风速10min
            'r46', # 平均风向30s
            'r61', # 机舱位置
            'r86', # 环境温度
            'r60', # 偏航夹角
            'r72','r73','r74', # 3个叶片桨距角实际值
            'r76','r78','r80', # 3个变桨电机温度值
            'r3', 'r59', # 风机状态，偏航状态机
            'r21' # 电网监测有功功率 Y
            ]

    # 数据读取
    df = data_read(file_name, cols)

    # 数据处理
    df = base_filter(df)
    df = radial_transform(df)
    df = temp_transform(df)
    df = yaw_error_abs(df)

    # 数据保存
    os.chdir(save_path)

    numerical_X_save(df)
    category_X_save(df)
    state_X_save(df)
    Y_save(df)








































