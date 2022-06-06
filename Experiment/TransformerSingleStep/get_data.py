import os
import sys
import time
import math
import pandas as pd
from pandas import read_csv
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 输入窗口
input_window = 100
# 预测窗口
output_window = 1

batch_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_inout_sequences(input_data, tw):
    """
        调用：train_sequence = create_inout_sequences(train_data, input_window)
        处理原始数据集得到模型的训练集，并转换为tensor
        每次从数据集中取出目标窗口(tw)长度的数据：(i,i+100)部分是输入，(i+1,i+100+1)部分是输出，即监督学习的对应的label
        最终得到训练集[ ([0-100],[1-101]), ([1-101],[2-102]), ..., ([1899-1999],[1900-2000]) ]，共1900个数据

    Args:
        input_data: 原始数据，即train_data
        tw: 输入窗口，即input_window = 100
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    # 转换成tensor
    return torch.FloatTensor(inout_seq)


def get_data():
    """
        导入CSV数据，对数据做归一化处理,提升模型的收敛速度,提升模型的精度
        初始化scaler在(-1,1)之间,然后使用scaler归一化数据，amplitude指序列振幅
        根据sampels将数据划分为数据集和测试集，调用create_inout_sequences()将其转换为tensor
    """

    # header=0，使用数据文件的第一行作为列名称，将第一列作为索引
    series = read_csv('./Experiment/data/Prometheus/minutedata.csv', header=0,
                      index_col=0, parse_dates=True, squeeze=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # reshape()更改数据的行列数，(-1, 1)将series变为一列 (2203,1)，归一化后再(-1)变为一行 (2203,)
    amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    # 反归一化：reamplitude = scaler.inverse_transform(amplitude.reshape(-1, 1)).reshape(-1)
    sampels = 2000
    # (2000,)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # view(-1)变成一行
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # test_data = torch.FloatTensor(test_data).view(-1)
    train_sequence = create_inout_sequences(train_data, input_window)
    # (1900,2,100)
    train_sequence = train_sequence[:-output_window]
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    # 把tensor移动到GPU上运行
    return train_sequence.to(device), test_data.to(device)


def get_batch(source, i, batch_size):
    """
        调用：data, targets = get_batch(train_data, i, batch_size)
        把源数据分为长度为batch_size的块，生成模型训练的输入和输入数据
    Args:
        source: 即train_data
        i: 每组数据从i开始,即当前batch的起始索引
        batch_size: 10
    """
    seq_len = min(batch_size, len(source) - 1 - i)
    # 每个batch的数据
    data = source[i:i+seq_len]
    # torch.stack(inputs, dim=?)→Tensor，对inputs(多个tensor)沿指定维度dim拼接，返回一维的tensor
    # 即source = [([0...100],[1...101]), ([1...101],[2...102])...]，取出item[0]拼接,即train_seq = [[0...100],[1...101]...]
    # torch.chunk(tensor, chunk_num, dim)将tensor在指定维度上(0行,1列)分为n块,返回一个tensor list,是一个tuple
    # 即将拼接后的source按每一列划分成一个tensor tuple
    input = torch.stack(torch.stack([item[0]
                        for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1]
                         for item in data]).chunk(input_window, 1))
    return input, target
