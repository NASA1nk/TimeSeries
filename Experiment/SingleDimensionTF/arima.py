import pmdarima as pm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 移动平均图，计算移动平均值和标准差
def draw_trend(ts, size):
    # 对size个数据进行移动平均
    rol_mean = ts.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = ts.rolling(window=size).std()
    # 查看原始数据的均值和方差
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    ax.plot(ts, c='blue', label='Original')
    ax.plot(rol_mean, c='red', label='Rolling Mean')
    ax.plot(rol_std, color="green", label="Rolling standard deviation")
    ax.legend(loc='best') 
    plt.savefig(f'./img/arima/arima_decompose_trend.png')


# Dickey-Fuller test
# 测试一个自回归模型是否存在单位根, 单位根是一个使得时间序列非平稳的一个特征，如果序列平稳，就不存在单位根，否则就会存在单位根
# 只要这个统计值是小于1%水平下的数字就可以极显著的拒绝原假设，认为数据平稳
# 注意，ADF值一般是负的，也有正的，但是它只有小于1%水平下的才能认为是及其显著的拒绝原假设
def test_stationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

    
# 移动平均法平滑处理
def draw_moving(ts, size):
    rol_mean = ts.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.DataFrame.ewm(ts, span=size).mean()
    # print(rol_weighted_mean)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    ax.plot(ts, c='blue', label='Original')
    ax.plot(rol_mean, c='red', label='Rolling Mean')
    ax.plot(rol_weighted_mean.to_numpy(), color="green", label="Weighted Rolling Mean")
    ax.legend(loc='best') 
    plt.savefig(f'./img/arima/arima_draw_moving.png')


# 分解趋势, 季节, 随机
def decompose(ts):
    # 返回包含三个部分 trend（趋势部分）, seasonal（季节性部分）和residual (残留部分)
    decomposition = seasonal_decompose(ts, period=1)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    f = plt.figure(figsize=(10, 6))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(4, 1, 1)
    ax.plot(ts, c='blue', label='Original')
    ax.legend()
    ax = f.add_subplot(4, 1, 2)
    ax.plot(trend, label='Trend')
    ax.legend()
    ax = f.add_subplot(4, 1, 3)
    ax.plot(seasonal, label='Seasonality')
    ax.legend()
    ax = f.add_subplot(4, 1, 4)
    ax.plot(residual, label='Residuals')
    ax.legend()
    plt.savefig(f'./img/arima/arima_decompose.png')
    return trend, seasonal, residual


# 数据平稳后，需要对模型定阶，即确定p、q的阶数
def draw_acf_pacf(ts, lags):
    f = plt.figure(figsize=(10,8), facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, ax=ax2,lags=lags)
    plt.savefig(f'./img/arima/arima_acf_pacf.png')


def arima(ts):
    model = sm.tsa.ARIMA(ts, order=(1,1,1))
    result_arima = model.fit()
    predict_ts = result_arima.predict()
    # # 一阶差分还原
    # diff_shift_ts = ts_diff_1.shift(1)
    # diff_recover_1 = predict_ts.add(diff_shift_ts)
    # # 再次一阶差分还原
    # rol_shift_ts = rol_mean.shift(1)
    # diff_recover = diff_recover_1.add(rol_shift_ts)
    # # 移动平均还原
    # rol_sum = ts_log.rolling(window=11).sum()
    # rol_recover = diff_recover*12 - rol_sum.shift(1)
    # 对数还原
    log_recover = np.exp(predict_ts)
    log_recover.dropna(inplace=True)
    return log_recover


if __name__ == "__main__":
    path = '../data/arima/kpi_normal_1.csv'
    df = pd.read_csv(path)
    series = df['value'].to_numpy()
    sample = df.shape[0]//10*7
    series = series[:sample]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)
    train_data = pd.Series(series)
    ts_log = np.log(train_data)
    # draw_trend(train_data, 100)
    # ret = test_stationarity(train_data)

    # 按t值好像是平稳的？
    # Test Statistic                -6.644628e+00
    # 接近0了
    # p-value                        5.303672e-09
    # #Lags Used                     6.600000e+01
    # Number of Observations Used    8.660000e+04
    # 1% ： 严格拒绝原假设； 5%： 拒绝原假设
    # Critical Value (1%)           -3.430426e+00
    # Critical Value (5%)           -2.861573e+00
    # Critical Value (10%)          -2.566788e+00
    # dtype: float64
    # print(ret)

    # draw_moving(ts_log, 12)

    # # 差分操作
    # # 12阶差分
    # diff_12 = ts_log.diff(12)
    # diff_12.dropna(inplace=True)
    # # 1阶差分
    # diff_12_1 = diff_12.diff(1)
    # diff_12_1.dropna(inplace=True)
    # ret = test_stationarity(diff_12_1)

    # Test Statistic                   -79.328859
    # p-value                            0.000000
    # #Lags Used                        66.000000
    # Number of Observations Used    86587.000000
    # Critical Value (1%)               -3.430426
    # Critical Value (5%)               -2.861573
    # Critical Value (10%)              -2.566788
    # dtype: float64
    # print(ret)

    # # 添加索引
    # t = list(df['timestamp'])
    # t = [datetime.datetime.fromtimestamp(i).strftime("%Y-%m-%d %H:%M:%S") for i in t]
    # df.index = pd.to_datetime(t)
    # trend, seasonal, residual = decompose(ts_log)
    # residual.dropna(inplace=True)
    # draw_trend(residual, 12)
    # ret = test_stationarity(residual)
    # print(ret)

    # 对于长期趋势成分采用1阶差分来进行处理
    # rol_mean = ts_log.rolling(window=10).mean()
    # ts_diff_1 = rol_mean.diff(1)
    # ts_diff_1.dropna(inplace=True)
    # ts_diff_2 = ts_diff_1.diff(1)
    # ts_diff_2.dropna(inplace=True)
    # draw_acf_pacf(ts_log, 30)

    ret = arima(ts_log)
    mse = sum((ret-train_data)**2)/train_data.size
    mae = sum(abs(ret-train_data))/train_data.size
    print(f'arima: {{MSE: {mse}, MAE: {mae}}}')
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    ax.plot(ret, c='blue', label='predict')
    ax.plot(train_data, color='red', label='Original')
    ax.set_title(f'MSE: {mse}, MAE: {mae}')
    ax.legend() 
    plt.savefig(f'./img/arima/arima_predict_scaler.png')
    
    f = plt.figure(figsize=(10, 6))
    f.patch.set_facecolor('white')
    ax = f.add_subplot(2, 1, 1)
    ax.plot(ret, c='blue', label='predict')
    ax.legend()
    ax = f.add_subplot(2, 1, 2)
    ax.plot(train_data, color='red', label='Original')
    ax.legend()
    plt.savefig(f'./img/arima/arima_predict_scaler_1.png')