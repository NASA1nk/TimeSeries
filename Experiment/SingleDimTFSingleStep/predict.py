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
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0,os.getcwd())


def predict(test_model, test_data, steps, scaler):
    ground_truth = test_data[:input_window+steps]
    test_model.eval()
    test_data = create_targets_sequences(test_data, input_window)
    test_data = test_data.to(device)
    # 取验证集的第一个数据,一步步往后预测
    data, _ = get_batch(test_data, 0, 1)
    with torch.no_grad():
        for i in range(steps):
            # 因为data是根据input_window划分来的,在第一次预测的时候,data[-input_window:]就是data,后续添加了预测结果后,只取input_window长度来预测
            output = test_model(data[-input_window:])
            # 拼接预测的结果,output[-1]即[0,100]的最后一个结果(1,1,1)，作为新的输入继续预测
            data = torch.cat((data, output[-1:]))
    # 最后拼接的data即(100,1,100+steps)
    data = data.cpu().view(-1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(data, c='red', linestyle='-.', label='predict')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.legend()
    plt.savefig(f'./img/predict/100_1_512_3_32_{steps}.png')

    # # 恢复数据
    # data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
    # ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # fig.patch.set_facecolor('white')
    # ax.plot(data, c='red', linestyle='-.', label='predict')
    # ax.plot(ground_truth, c='blue', label='ground_truth')
    # ax.legend()
    # plt.savefig(f'./Experiment/SingleDimTFSingleStep/img/predict_normal_{steps}_{epoch}.png')


def plot_loss(eval_model, val_data, scaler):
    eval_model.eval()
    mse_loss_list = []
    mae_loss_list = []
    diff = []
    predict = torch.Tensor(0)
    ground_truth = torch.Tensor(0)
    with torch.no_grad():
        # batch size = 1
        for i in range(0, len(val_data)-1):
            data, targets = get_batch(val_data, i, 1)
            output = eval_model(data)
            mse = nn.MSELoss()
            mae = nn.L1Loss()
            mse_loss = mse(output, targets).cpu().item()
            mae_loss = mae(output, targets).cpu().item()
            mse_loss_list.append(mse_loss)
            mae_loss_list.append(mae_loss)
            diff.append(abs((output[-1].view(-1).cpu() - targets[-1].view(-1).cpu()).item()))
            predict = torch.cat((predict, output[-1].view(-1).cpu()), 0)
            ground_truth = torch.cat((ground_truth, targets[-1].view(-1).cpu()), 0)
            # writer.add_scalar('mse_loss', mse_loss, i)
            # writer.add_scalar('mae_loss', mae_loss, i)
    p99 = np.percentile(diff, 99).item()
    marker = []
    for i, v in enumerate(diff):
        if v >= p99:
            marker.append(i)
            print(i, diff[i])
    # 恢复数据
    predict = scaler.inverse_transform(predict.reshape(-1,1)).reshape(-1)
    ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    # linewidth默认值1.5
    
    ax.plot(ground_truth, c='blue', linewidth=1, marker='o', markerfacecolor = 'black', markevery=marker, label='ground_truth')
    ax.plot(predict, c='red', linewidth=1, label='predict')
    ax.plot(predict-ground_truth, c="green", label="diff")
    ax.legend() 
    plt.savefig(f'./img/100_1_512_1_32/predict_all.png')



    # print(len(marker))
    # fig, ax = plt.subplots(1, 1, figsize=(40, 5))
    # fig.patch.set_facecolor('white')
    # ax.plot(mse_loss_list, c='blue', label='MSE')
    # ax.plot(mae_loss_list, c='red', label='MAE')
    # ax.plot(diff, color="green", label="diff", marker='o', markerfacecolor = 'r', markevery=marker)
    # ax.legend() 
    # plt.savefig(f'./img/100_1_512_1_32/loss.png')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = '../data/2018AIOpsData/kpi_normal_1.csv'
    # data_path = '../data/2018AIOpsData/kpi_12.csv'
    # data_path = './Experiment/data/2018AIOpsData/kpi_12.csv'
    # test_data, scaler = get_test_data(data_path)
    _, _, test_data, scaler = get_data(data_path)
    test_data = test_data.to(device)
    feature = 512
    layers = 3
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    # 恢复模型, 将model中的参数加载到new_model中       
    # model_path = './Experiment/SingleDimTFSingleStep/best_model/100_1_512_1_32.pth' 
    model_path = './best_model/100_1_512_3_32.pth' 
    # model_path = './best_model/100_1_512_1_32_adam.pth' 
    name = model_path.split('/')[-1][:-4]
    # writer = SummaryWriter(comment=f'./loss/{name}_new_data', flush_secs=20)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 预测
    # input_window = 100
    # steps = len(test_data) - input_window
    # for steps in tqdm(range(2, 100)):
        # predict(model, test_data, steps, scaler)
    plot_loss(model, test_data, scaler)