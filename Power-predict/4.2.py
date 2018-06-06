
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


if __name__ == '__main__':

    os.chdir(r'C:\Users\Chinawindey\Desktop\新建文件夹 (2)')
    y = pd.read_pickle('y.pkl')
    X_cat = pd.read_pickle('X_cat.pkl')
    X_num = pd.read_pickle('X_num.pkl')
    X_dum = pd.read_pickle('X_dum.pkl')


    df = pd.concat([y, X_num, X_cat, X_dum],axis=1)

    df = df.dropna(how='any', axis=0)  # 删除所有为0的列

    # 划分训练集与测试集
    df_train = df['2016-01-01':'2016-12-31']
    df_test = df['2017-01-01':'2017-12-31']

    X_train = np.array(df_train.ix[:, 1:])
    y_train = np.array(df_train.ix[:, 0])
    X_test = np.array(df_test.ix[:, 1:])
    y_test = np.array(df_test.ix[:, 0])

    model = xgboost.XGBRegressor(max_depth=3,
                                 learning_rate=0.8,
                                 n_estimators=200,
                                 silent=False,
                                 objective='reg:linear',
                                 nthread=-1,
                                 gamma=10,
                                 min_child_weight=5,
                                 max_delta_step=0,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 reg_alpha=0,
                                 reg_lambda=0,
                                 seed=123,
                                 missing=None)


    print(cross_val_score(model, X_train, y_train, scoring='r2', cv=5))

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(r2_score(y_train, y_train_pred))
    print(r2_score(y_test, y_test_pred))

    # 训练集作图
    plt.subplot(2, 1, 1)
    plt.plot(df_train.index, y_train, label='real power')
    plt.plot(df_train.index, y_train_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(df_train.index, y_train - y_train_pred, label='error')
    plt.legend()
    plt.show()

    # 预测并作图
    plt.subplot(2, 1, 1)
    plt.plot(df_test.index, y_test, label='real power')
    plt.plot(df_test.index, y_test_pred, label='predict power')
    plt.subplot(2, 1, 2)
    plt.plot(df_test.index, y_test - y_test_pred, label='error')
    plt.legend()
    plt.show()























