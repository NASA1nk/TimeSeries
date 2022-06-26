#-*- encoding: utf-8 -*-
# test
import numpy as np
import time
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

def diff_smooth(ts, interval):
    """平滑处理"""
    # interval最小单位为秒，除以60就变为分钟，方便下面处理.
    wide = interval/60
    # 一阶差分
    dif = ts.diff().dropna()
    # 描述性统计得到：min，25%，50%，75%，max值
    td = dif.describe()
    # 定义高点阈值，1.5倍四分位距之外
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    # 定义低点阈值
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    i = 0
    # 变化幅度超过阈值的点的索引
    forbid_index = dif[(dif > high) | (dif < low)].index
    while i < len(forbid_index) - 1:
        n = 1
        # 异常点的起始
        start = forbid_index[i]
        while forbid_index[i+n] == start + datetime.timedelta(minutes=n):
            n += 1
        i += n - 1
        # 异常点的结束
        end = forbid_index[i]
        # np.linspace(start, end, num)生成等差数列
        # 用前后值均匀填充
        value = np.linspace(ts[start - datetime.timedelta(minutes=wide)], ts[end + datetime.timedelta(minutes=wide)], n)
        ts[start: end] = value
        i += 1
    return ts

def data_to_datetimeindex(timestamp, value):
    """将数据变为Series类型"""
    time_list = []
    for i in range(len(timestamp)):
        a = time.localtime(timestamp[i])
        b = time.strftime("%Y-%m-%d %H:%M:%S", a)
        time_list.append(b)

    dta = pd.Series(value)
    dta = dta.fillna(dta.mean())
    dta.index = pd.Index(time_list)
    dta.index = pd.DatetimeIndex(dta.index)
    return dta



def get_train_data(data_dir, predict_time):
    """从csv文件中获取训练数据
    返回：
    dta: Series类型，时间序列
    timestamp_list: 时间戳列表
    value_list: 数据列表
    """
    filename = data_dir
    data = pd.read_csv(filename)
    timestamp_list = []
    value_list = []
    # data中去掉后面predict_time数据，作为验证部分
    data = data[:-int(predict_time)]
    for timestamp, value in zip(data['timestamp'],data['value']):
        timestamp_list.append(timestamp)
        value_list.append(value)

    dta = data_to_datetimeindex(timestamp_list, value_list)
    return dta, timestamp_list, value_list

def get_truth_data(data_dir, predict_time):
    """从csv文件中获取预测真实数据
        返回：
        dta: Series类型，时间序列
        timestamp_list: 时间戳列表
        value_list: 数据列表
        """
    filename = data_dir
    data = pd.read_csv(filename)
    value_list = []
    # data中只保留后面predict_time数据，作为验证
    data = data[-int(predict_time):]
    for _, value in zip(data['timestamp'], data['value']):
        value_list.append(value)

    return value_list

class ARIMAModel(object):
    def __init__(self, predict_time):
        self.predict_time = predict_time
        pass

    def train(self, dta, x, y):
        """
        for i in range(0, len(mydata_tmp)):
            mydata_tmp[i] = math.log(mydata_tmp[i])
        """
        """
        p为ARMA模型的参数，一般p去小于length/10的数
        但是由于数据的问题，所以分情况设置
        """
        res = sm.tsa.arma_order_select_ic(dta, max_ar=7, max_ma=0,ic=['bic'],trend='c')
        p = res.bic_min_order[0] # trend must be one of: 'n', 'c'
        q = res.bic_min_order[1]
        # 建立ARMA模型
        # freq为时间序列的偏移量
        try:
            try:
                model_tmp = ARIMA(dta, order=(p,1,q))
                # method为css-mle
                #model = model_tmp.fit(disp=-1)
                model = model_tmp.fit(disp=-1, method='mle')
                dir(model)
                return model
            except:
                model_tmp = ARIMA(dta, order=(1,1,1))
                model = model_tmp.fit(disp=-1, method='mle')
                dir(model)
                return model
        except:
            return
    
    def predict(self, model, y):
        predict_outcome = model.forecast(self.predict_time)
        return predict_outcome[0]

if __name__ == "__main__":
    data_dir = './Experiment/data/2018AIOpsData/kpi_normal_1.csv'
    df = pd.read_csv(data_dir, header=0, parse_dates=True, squeeze=True)
    series = df['value']
    sample = df.shape[0]-df.shape[0]//10*9
    predict_time = sample
    print(predict_time)
    ori_data, timestamp_list, value_list = get_train_data(data_dir, predict_time)

    create_model = ARIMAModel(predict_time)
    train_model = create_model.train(ori_data, timestamp_list, value_list)
    print(dir(create_model))
    print(type(create_model))
    if train_model is not None:
        predict_data = create_model.predict(train_model, value_list)
        print("the prediction result:")
        print(predict_data)
    else:
        print('None')
