import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

# date	    store_nbr	sales	    onpromotion
# 2013/1/2	1	        7417.148	0
data = pd.read_csv(r'1.csv').iloc[1:, 2:3].values

# 训练窗口为5，预测窗口为3
def data_process(data, window_size, predict_size):

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    data_in = []
    data_out = []
    for i in range(data.shape[0]-window_size-predict_size):
        data_in.append(data[i:i+window_size].reshape(1, window_size)[0])
        data_out.append(data[i+window_size:i+window_size+predict_size].reshape(1, predict_size)[0])
    data_in = np.array(data_in).reshape(-1, window_size)
    data_out = np.array(data_out).reshape(-1, predict_size)
    # 将模型输入与输出分开保存在字典中
    data_process = {
        'datain': data_in,
        'dataout': data_out
    }
    return data_process


data_prepro = data_process(data, 30, 7)
X_train, X_test, y_train, y_test = train_test_split(data_prepro['datain'],
                                                    data_prepro['dataout'],
                                                    test_size=0.2)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))




train_data = Data.TensorDataset(X_train, y_train)
test_data = Data.TensorDataset(X_test, y_test)
train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, drop_last=True)