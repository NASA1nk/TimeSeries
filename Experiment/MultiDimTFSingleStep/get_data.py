import os
import sys
import time
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

input_window = 100
output_window = 1

# [0:95,0,0,0,0,0] -> [0,99] 还是 [0,95,0,0,0,0,0] -> [5,105]
def create_steps_sequences(source_data, input_window, output_window):
    targets = []
    L = len(source_data)
    for i in range(L-input_window):
        # 单维
        # seq = source_data[i:i+input_window-output_window]
        # train_seq = np.append(seq, [0] * output_window)
        # train_label = source_data[i:i+input_window]
        # # train_label = source_data[i+output_window:i+output_window+input_window]
        # 多维
        # seq = source_data[i:i+input_window, :][:-output_window, :]
        seq = source_data[i:i+input_window-output_window, :]
        # train_seq = np.append(seq, np.zeros((output_window,10)), axis=0)
        train_label = source_data[i:i+input_window,:]
        targets.append((train_seq, train_label))
    # 转换成tensor
    return torch.FloatTensor(targets)