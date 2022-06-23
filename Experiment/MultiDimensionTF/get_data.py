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
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


def create_targets_sequences(source_data, input_window, output_window):
    targets = []
    L = len(source_data)
    for i in range(L - input_window + 1 - output_window):
        train_seq = source_data[i:i+input_window]
        train_label = source_data[i+output_window:i+input_window+output_window]
        targets.append((train_seq, train_label))
    return torch.FloatTensor(targets)


def get_data(path, input_window, output_window):
    df = pd.read_excel(path, 'Sheet1', parse_dates=["date"])
    data = df.loc[(df["date"] >= pd.Timestamp(date(2014, 1, 1))) & (df["date"] <= pd.Timestamp(date(2014, 2, 10)))]
    # df = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = data.loc[:, "MT_200":  "MT_209"].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series)
    sample1 = data.shape[0]//10*7
    sample2 = data.shape[0]//10*9
    train_data = amplitude[:sample1]
    val_data = amplitude[sample1:sample2]
    test_data = amplitude[sample2:]

    train_data = create_targets_sequences(train_data, input_window, output_window)
    val_data = create_targets_sequences(val_data, input_window, output_window)
    test_data = create_targets_sequences(test_data, input_window, output_window)
    return train_data, val_data, test_data, scaler


def get_batch(source, i, batch_size, input_window, feature):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    # (100,32,10)
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    # input_r = input.repeat(input.shape[2], feature/input.shape[2])
    # target_r = target.repeat(input.shape[2], feature/input.shape[2])
    # F.interpolate(input, scale_factor=feature/input.shape[2])
    return input, target