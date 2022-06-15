import os
import sys
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from get_data import create_targets_sequences, get_data, get_batch
from models import TransformerModel
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0,os.getcwd())


# 使用测试集和best model来预测后n步
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
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    fig.patch.set_facecolor('white')
    ax.plot(data, c='red', linestyle='-.', label='predict')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.legend()
    plt.savefig(f'./Experiment/SingleDimTFSingleStep/img/predict/100_1_512_3_32_{steps}.png')

    # # 恢复数据
    # data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
    # ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # fig.patch.set_facecolor('white')
    # ax.plot(data, c='red', linestyle='-.', label='predict')
    # ax.plot(ground_truth, c='blue', label='ground_truth')
    # ax.legend()
    # plt.savefig(f'./Experiment/SingleDimTFSingleStep/img/predict_normal_{steps}_{epoch}.png')

if __name__ == "__main__":
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = './Experiment/data/2018AIOpsData/kpi_normal_1.csv'
    _, _, test_data, scaler = get_data(data_path)
    feature = 512
    layers = 3
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    # 恢复模型, 将model中的参数加载到new_model中       
    model_path = './Experiment/SingleDimTFSingleStep/best_model/100_1_512_3_32.pth'     
    model.load_state_dict(torch.load(model_path))  
    # 预测
    input_window = 100
    # steps = len(test_data) - input_window
    # steps = 20
    for steps in tqdm(range(2, 100)):
        predict(model, test_data, steps, scaler) 