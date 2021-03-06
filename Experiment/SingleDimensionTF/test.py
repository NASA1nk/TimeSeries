import os
import sys
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from get_data import create_targets_sequences, get_data, get_batch, get_test_data
from models import TransformerModel
from tqdm import tqdm

sys.path.insert(0,os.getcwd())


def predict(test_model, test_data, input_window, output_window, steps, scaler):
    ground_truth = test_data[:input_window+steps]
    test_model.eval()
    test_data = create_targets_sequences(test_data, input_window, output_window).to(device)
    # 取验证集的第一个数据,一步步往后预测
    data, _ = get_batch(test_data, 0, 1)
    with torch.no_grad():
        for i in range(steps):
            # 因为data是根据input_window划分来的,在第一次预测的时候,data[-input_window:]就是data,后续添加了预测结果后,只取input_window长度来预测
            output = test_model(data[-input_window:])
            # 拼接预测的结果,output[-1]即[0,100]的最后一个结果(1,1,1)，作为新的输入继续预测
            data = torch.cat((data, output[-output_window:]))
    # 最后拼接的data即(100,1,100+steps)
    data = data.cpu().view(-1)
    data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
    ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(data, c='red', linestyle='-.', label='predict')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.legend()
    plt.savefig(f'./img/predict_{steps}.png')


def plot_diff(test_model, test_data, scaler, input_window, output_window):
    test_model.eval()
    predict = torch.Tensor(0)
    ground_truth = torch.Tensor(0)
    diff_list = []
    mse = 0.
    mae = 0.
    with torch.no_grad():
        for i in range(0, len(test_data)-1):
            data, targets = get_batch(test_data, i, 1, input_window)
            output = test_model(data)
            # 一个元素的tensor可以通过item()获取标量 
            diff = nn.L1Loss()(targets[-output_window:], output[-output_window:]).cpu().item() * output_window
            diff2 = nn.MSELoss()(targets[-output_window:], output[-output_window:]).cpu().item() * output_window
            diff_list.append(diff)
            mae += diff
            mse += diff2
            predict = torch.cat((predict, output[-output_window:].view(-1).cpu()), 0)
            ground_truth = torch.cat((ground_truth, targets[-output_window:].view(-1).cpu()), 0)
    mse = mse / i
    mae = mae / i
    # 记录超过p99的值，标记在图上
    p99 = np.percentile(diff_list, 99).item()
    marker = []
    for i, v in enumerate(diff_list):
        if v >= p99:
            marker.append(i)
            # print(i, diff_list[i])
    # 恢复数据
    predict = scaler.inverse_transform(predict.reshape(-1,1)).reshape(-1)
    ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    mse_sc = 0.
    mae_sc = 0.
    for y1, y2 in zip(predict, ground_truth):
        dif = abs(y1-y2)
        mae_sc += dif
        mse_sc += dif*dif
    mse_sc /= len(predict)
    mae_sc /= len(predict)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.plot(predict, c='red', marker='o', markerfacecolor='black', markevery=marker, label='predict')
    # ax.plot(diff_list, c="green", label="diff")
    ax.legend() 
    plt.savefig(f'./Experiment/SingleDimensionTF/img/{input_window}_{output_window}_512_{layers}_32_adam_predict.png')
    # plt.savefig(f'./img/{input_window}_{output_window}_512_{layers}_32_adam_predict.png')
    return mse, mae, mse_sc, mae_sc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_path = './Experiment/data/2018AIOpsData/kpi_normal_1.csv'
    # data_path = '../data/2018AIOpsData/kpi_normal_1.csv'
    data_path = './Experiment/data/2018AIOpsData/kpi_12.csv'
    # data_path = '../data/2018AIOpsData/kpi_12.csv'
    input_window = 200
    output_window = 5
    test_data, scaler = get_test_data(data_path, input_window, output_window)
    # _, _, test_data, scaler = get_data(data_path, input_window, output_window)
    test_data = test_data.to(device)
    feature = 512
    layers = 1
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    # 恢复模型, 将model中的参数加载到new_model中   
    model_path = './Experiment/SingleDimensionTF/best_model/200_5_512_1_64.pth'
    # model_path = './best_model/5_1_512_1_32.pth' 
    name = model_path.split('/')[-1][:-4]
    model.load_state_dict(torch.load(model_path, map_location=device))
    mse, mae, mse_sc, mae_sc = plot_diff(model, test_data, scaler, input_window, output_window)
    print(f'{name}: {{mse: {mse}, mae: {mae}}} {{{mse_sc} {mae_sc}}}')


    