# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :         Zeke
   date：           2018/5/30
   Description :    数据筛选
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import os
import math



if __name__ == '__main__':

    # 定义文件路径
    os.chdir(r'D:\00_工作日志\O\2018-06\赤峰项目故障\data\赤峰二期5min数据')
    file_name = 'CFEQ_wt1_5min_201601-201712.csv'

    save_path = r'C:\Users\Chinawindey\Desktop\新建文件夹 (2)'
    save_name = '123.csv'

    # 筛选的变量
    col_name = ['r1', # 时间
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


    # 定义训练集时间
    ds = '2016-01-15'
    dt = '2016-04-15'

    # 定义预测集时间
    ds_fu = '2016-04-16'
    dt_fu = '2016-04-30'


    # 读取文件到DataFrame
    with open(file_name, 'r') as f:  # 读取文件进入
        df = pd.read_csv(f, low_memory=False)

    # 切片选择的列
    df = df.loc[:, col_name]

    # 取值范围筛选
    df[(df['r21'] < 5) & (df['r21'] > 2200)] = np.nan # 功率异常值
    df[(df['r43'] < 0.05) & (df['r43'] > 50)] = np.nan # 筛选掉风速大于50
    df[(df['r46'] < 0) & (df['r46'] > 360)] = np.nan  # 筛选掉风向在360°以外
    df[(df['r86'] < -50) & (df['r86'] > 50)] = np.nan # 筛选掉机舱外温度在50度以上
    df[(df['r60'] < -180) & (df['r60'] > 180)] = np.nan  # 筛选掉风向在360°以外
    df[(df['r72'] < -5) & (df['r72'] > 92)] = np.nan  # 筛选桨距角1
    df[(df['r73'] < -5) & (df['r73'] > 92)] = np.nan  # 筛选桨距角2
    df[(df['r74'] < -5) & (df['r74'] > 92)] = np.nan  # 筛选桨距角3
    df[(df['r76'] < -5) & (df['r76'] > 92)] = np.nan  # 筛选桨距角3

    # index改为时间
    df.index = df.loc[:, 'r1']
    df.index = pd.to_datetime(df.index)
    print(df.head())

    # 删除所有带空值的行
    df = df.dropna(how='any', axis=0)

    # 风向，机舱位置增加sin与cos值
    df['r46_cos'] = df.loc[:, 'r46'].apply(lambda x: math.cos(math.radians(x)))
    df['r46_sin'] = df.loc[:, 'r46'].apply(lambda x: math.sin(math.radians(x)))

    df['r61_cos'] = df.loc[:, 'r61'].apply(lambda x: math.cos(math.radians(x)))
    df['r61_sin'] = df.loc[:, 'r61'].apply(lambda x: math.sin(math.radians(x)))

    del df['r46']
    del df['r61']

    # 偏航误差改成绝对值
    df['r60'] = np.abs(df['r60'])

    # 温度更改成绝对温度
    df['r86'] = df['r86'] + 273.15
    df['r76'] = df['r76'] + 273.15
    df['r78'] = df['r78'] + 273.15
    df['r80'] = df['r80'] + 273.15






    # 增加季节与月份哑向量
    df['month'] = df.loc[:, 'r1'].apply(lambda x: int(x[5:7]))
    df['season'] = (df.loc[:, 'month'] / 4 + 1).apply(math.floor)
    df1 = pd.get_dummies(df['month'], prefix='month')
    df2 = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, df1, df2], axis=1)

    print(df.tail())

    # 保存文件
    os.chdir(save_path)
    df.to_csv(save_name)




