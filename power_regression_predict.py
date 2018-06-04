import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor



def data_process(file_name, col_name, ds, dt):
    # "对原始数据进行删除/填充/切割/变化等"

    with open(file_name, 'r') as f: # 读取文件进入dataframe
        df = pd.read_csv(f,header=0,index_col=0,low_memory=False)

    #防止文件中有空格，将空格替换为NaN,并且将表格转换成数值
    df = df.replace(' ', np.NaN)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    #切片选择的列,并且以5min填充行
    df = df.loc[:, col_name]

    #切片训练时间
    df.index = pd.to_datetime(df.index)
    # df = df.resample('10min',label='left').first()
    df = df.loc[ds:dt, :]

    #删除空值
    df = df.dropna(axis=0, how='any') #删除表中含有NaN的行
    #data = data.fillna(df.median(),inplace=True) #将空值填充为每一列的中值

    #筛选功率大于5的行，筛选掉停机工况
    df['r18'] = df['r18'].apply(float)
    df = df[df['r18'] > 5]
    df = df[df['r18'] < 2100]

    return df


def svr_model(df, df_fu):
    # 支持向量机
    estimator = SVR(kernel='rbf')
    param_grid = {'C': np.logspace(-2,0,10), 'gamma': list(range(1,10,1))}
    clf = GridSearchCV(estimator, param_grid, scoring='r2', verbose=2, refit=True, cv=5, n_jobs=-1)
    clf.fit(df.values[:, :-1], df.values[:, -1])
    print(clf.best_params_, clf.best_score_)
    print(clf.score(df_fu.values[:, :-1], df_fu.values[:, -1]))  # 测试在测试集上的R2

    # 调用画图
    fig_error(clf, df, df_fu)


def random_forest_model(df, df_fu):
    # 随机森林
    estimator = RandomForestRegressor(max_features='sqrt', random_state=123)
    param_grid = {'n_estimators': list(range(10, 500, 20)), 'max_depth': [1, 2]}
    clf = GridSearchCV(estimator, param_grid, scoring='r2', verbose=2, refit=True, cv=5, n_jobs=-1)
    clf.fit(df.values[:, :-1], df.values[:, -1])
    print(clf.best_params_, clf.best_score_)
    print(clf.score(df_fu.values[:, :-1], df_fu.values[:, -1]))  # 测试在测试集上的R2
    # 调用画图
    fig_error(clf, df, df_fu)


def fig_error(clf, df, df_fu):
    # 训练集作图
    df_y_pred = clf.predict(df.values[:, :-1])
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df.values[:, -1], label='real power')
    plt.plot(df.index, df_y_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df.values[:, -1] - df_y_pred, label='error')
    plt.legend()
    plt.show()

    # 预测并作图
    df_fu_y_pred = clf.predict(df_fu.values[:, :-1])
    plt.subplot(2, 1, 1)
    plt.plot(df_fu.index, df_fu.values[:, -1], label='real power')
    plt.plot(df_fu.index, df_fu_y_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(df_fu.index, df_fu.values[:, -1] - df_fu_y_pred, label='error')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # 读取数据集
    os.chdir(r'C:\Users\Zeke\Desktop\download\ZYBL_wt1_5min_2016')

    file_name = 'ZYBL_wt1_5min_2016.csv'

    col_name = ['r72','r73','r74','r47','r76','r78','r80','r43','r44','r45','r18']

    ds = '2016-01-15'
    dt = '2016-08-31'

    #定义预测的时间
    ds_fu = '2016-09-01'
    dt_fu = '2016-12-31'


    df = data_process(file_name, col_name, ds, dt)
    #缩放x到[0,1]
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler_space = min_max_scaler.fit_transform(df.values)
    df = pd.DataFrame(data=scaler_space,index=df.index,columns=df.columns)


    df_fu = data_process(file_name, col_name, ds_fu, dt_fu)
    #缩放x到[0,1]
    scaler_fu = min_max_scaler.transform(df_fu.values)
    df_fu = pd.DataFrame(data=scaler_fu,index=df_fu.index,columns=df_fu.columns)

    # 调用随机森林
    # random_forest_model(df, df_fu)

    #调用svm
    svr_model(df, df_fu)





