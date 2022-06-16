import os
import sys
import time
import math
import pandas as pd
import numpy as np
from datetime import date
from matplotlib import pyplot
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def create_targets_sequences(source_data, input_window, output_window):
    """
        处理原始数据得到模型的训练集
        每次从数据集中取出输入窗口(input_window)长度的数据：(i,i+100)部分是输入，(i+1,i+100+1)部分是输出
        最终得到训练集[ ([0-100],[1-101]), ([1-101],[2-102]), ..., ([1899-1999],[1900-2000]) ]，共1900个数据

    Args:
        source_data: 需要转换的原始数据
        input_window: 输入窗口
    """
    targets = []
    L = len(source_data)
    for i in range(L-input_window):
        train_seq = source_data[i:i+input_window]
        train_label = source_data[i+output_window:i+input_window+output_window]
        targets.append((train_seq, train_label))
    return torch.FloatTensor(targets)


def get_data(path, input_window, output_window):
    """
        导入CSV数据, 对数据做归一化处理, 初始化scaler在(-1,1)之间,然后使用scaler归一化数据, amplitude指序列振幅
        根据sample将数据划分为数据集和测试集, 调用create_targets_sequences处理输入输出, 并将其转换为tensor
    """
    # header=0，使用数据文件的第一行作为列名称，将第一列作为索引
    df = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    # series = df.to_numpy()
    # 通过df.loc来限制行列
    series = df['value'].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit_transform要求数据形状
    # reshape()更改数据的行列数，(-1, 1)将df变为一列 (2203,1)，归一化后再(-1)变为一行 (2203,)
    amplitude = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)
    sample1 = 90000
    sample2 = 110000
    train_data = amplitude[:sample1]
    val_data = amplitude[sample1:sample2]
    test_data = amplitude[sample2:]

    # view(-1)变成一行
    # train_sequence即(train_data-input_window, 2, input_window)
    train_data = create_targets_sequences(train_data, input_window, output_window)
    # 剔除output_window = 1个元素,即[[99899,99998], [99900, 99999]]
    # train_data = train_data[:-output_window]
    val_data = create_targets_sequences(val_data, input_window, output_window)
    # val_data = val_data[:-output_window]
    test_data = create_targets_sequences(test_data, input_window, output_window)
    return train_data, val_data, test_data, scaler


def get_batch(source, i, batch_size, input_window):
    """
        调用：data, targets = get_batch(train_data, i, batch_size)
        把源数据分为长度为batch_size的块，生成模型训练的输入和输入数据
    """
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # torch.stack(inputs, dim=?)→Tensor, 对inputs(多个tensor)沿指定维度dim拼接，返回一个大的的tensor
    # torch.chunk(tensor, chunk_num, dim), 将tensor在指定维度上(0行,1列)分为n块,返回一个tensor tuple
    # 即将拼接后的source按每一列划分成一个tensor tuple
    input = torch.stack(torch.stack([item[0]
                        for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1]
                         for item in data]).chunk(input_window, 1))
    return input, target


def get_test_data(path, input_window):
    """
    获取新数据全部作为测试集
    """
    df = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    series = df['value'].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)
    test_data = create_targets_sequences(amplitude, input_window)
    return test_data, scaler