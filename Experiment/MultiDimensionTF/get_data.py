import os
import sys
import time
import math
from datetime import date
import pandas as pd
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# [0:95,0,0,0,0,0] -> [0,99] 还是 [0,100] -> [5,105]
def create_inout_sequences(source_data, input_window, output_window):
    targets = []
    L = len(source_data)
    for i in range(L-input_window):
        # 1. 每次窗口大小是input_window:
        # src: [0:95,0,0,0,0,0]即，label: [0,99] 
        seq = source_data[i:i+input_window-output_window, :]
        train_seq = np.append(seq, np.zeros((output_window,10)), axis=0)
        
        train_label = source_data[i:i+input_window, :]
        # [0,100] -> [5,105]，数据个数会改变
        
        # train_label = source_data[i+output_window:i+input_window+output_window, :]
        targets.append((train_seq, train_label))
    return torch.FloatTensor(targets)


def get_data(path, input_window, output_window):
    df = pd.read_excel(path, 'Sheet1', parse_dates=["date"])
    data = df.loc[(df["date"] >= pd.Timestamp(date(2014, 1, 1))) & (df["date"] <= pd.Timestamp(date(2014, 2, 10)))]
    
    # df = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    data = data.loc[:, "MT_200":  "MT_209"].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(data)
    samples = 2800
    train_data = amplitude[:samples]
    test_data = amplitude[samples:]

    train_sequence = create_inout_sequences(train_data,input_window, output_window)
    train_sequence = train_sequence[:-output_window]
    test_data = create_inout_sequences(test_data,input_window, output_window)
    test_data = test_data[:-output_window]
    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)).squeeze()
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1)).squeeze()
    return input, target