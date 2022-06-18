import os
import sys
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from get_data import get_data, get_batch
from models import TransformerModel
from test import plot_diff
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.getcwd())
torch.manual_seed(0)
np.random.seed(0)


def train(train_data, batch_size, input_window):
    # 设置模型为trainning模式,启用BatchNormalization和Dropout
    model.train()
    # 一个epoch总的损失
    total_loss = 0.
    avg_loss = 0.
    start_time = time.time()
    # 根据每次划分得到的i获取每一个batch的数据
    for batch, i in enumerate(range(0, len(train_data)-1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size, input_window)
        # 反向传播前将梯度清零，即将loss关于weight的导数变成0
        optimizer.zero_grad()
        # 前向传播，即把数据输入网络（调模型中的forward函数）中并得到输出
        output = model(data)
        loss = criterion(output, targets)
        # 反向传播梯度
        loss.backward()
        # 梯度裁剪：在BP过程中会产生梯度消失（偏导无限接近0）解决方法是设定一个阈值，当梯度小于阈值时更新的梯度为阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        # 根据梯度反向传播来更新网络参数（通常用在每个batch中，应该在train()中，只有这样模型才会更新）
        optimizer.step()
        # 获取loss的标量，item():得到一个元素张量里面的元素值，即将一个零维张量转换成浮点数
        total_loss += loss.item()
        avg_loss += len(data[0]) * loss.item()
        # 把整个epoch分为5部分，分别打印训练信息
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            # 这部分轮训练花费的时间
            cost_time = time.time() - start_time
            # 第几个epoch，当前的batch数目/一个epoch的总batch数，学习率，训练花费时间，MSEloss
            print('-' * 75)
            print('| epoch {:2d} | {:5d} / {:} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} |'.format(
                                                    epoch, 
                                                    batch, len(train_data) // batch_size,
                                                    scheduler.get_last_lr()[0],
                                                    cost_time / log_interval * 1000,
                                                    cur_loss))
            total_loss = 0
            start_time = time.time()
    avg_loss = avg_loss / len(train_data)
    writer.add_scalar('train_loss', avg_loss, epoch) 
    

# 使用验证集来评估每个epoch训练后的模型
def evaluate(eval_model, val_data, batch_size, input_window):
    # 设置为evaluation模式，不启用BatchNormalization和Dropout，将BatchNormalization和Dropout置为False（即training的属性置为False）
    eval_model.eval()
    total_loss = 0.
    # 表明当前计算不需要反向传播, 模型的评估和模型的训练逻辑基本相同, 唯一的区别是评估只需要forward pass, 不需要backward pass
    with torch.no_grad():
        for i in range(0, len(val_data)-1, batch_size):
            data, targets = get_batch(val_data, i, batch_size, input_window)
            output = eval_model(data)
            # 没有反向传播，可以直接获取loss(是一个tensor)的标量item()
            loss = criterion(output, targets).cpu().item()
            # 以batch为单位计算的loss, 所以乘batch size恢复(最后一次不足一个batch,所以是data[0])
            total_loss += len(data[0]) * loss
    # 返回整个验证集的所有元素的平均MSEloss
    avg_loss = total_loss / len(val_data)
    # 记录关键指标, 保存在本地
    writer.add_scalar('eval_loss', avg_loss, epoch)
    return avg_loss


# 使用验证集可视化模型的loss
def plot_loss(eval_model, val_data, batch_size, scaler, input_window):
    eval_model.eval()
    total_loss = 0.
    predict = torch.Tensor(0)
    ground_truth = torch.Tensor(0)
    with torch.no_grad():
        # batch_size = 1 
        for i in range(0, len(val_data)-1):
            data, targets = get_batch(val_data, i, 1, input_window)
            output = eval_model(data)
            loss = criterion(output, targets).cpu().item()
            total_loss += loss
            # torch.cat()将两个tensor拼接在一起，指定拼接的维数dim可以不同，其余维数要相同，二维的0表示按行拼接，1表示按列拼接
            # 这里是拼接每个batch中，网络的输出列表和当前标签label列表的最后一个值
            predict = torch.cat((predict, output[-1].view(-1).cpu()), 0)
            ground_truth = torch.cat((ground_truth, targets[-1].view(-1).cpu()), 0)
    # 恢复数据
    predict = scaler.inverse_transform(predict.reshape(-1,1)).reshape(-1)
    ground_truth = scaler.inverse_transform(ground_truth.reshape(-1,1)).reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.plot(predict, c='red', label='predict')
    ax.plot(predict-ground_truth, color="green", label="diff")
    ax.legend() 
    plt.savefig(f'./img/100_1_512_2_32_adam/Epoch_{epoch}.png')
    # plt.savefig(f'./img/50_1_512_1_32/Epoch_{epoch}_loss.png')
    # 返回验证集所有数据的平均MSEloss
    return total_loss / i


if __name__ == "__main__":
    start_time = time.time()
    # 指定device，后续可以调用to(device)把Tensor迁移到device上
    # torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path = './Experiment/data/2018AIOpsData/kpi_normal_1.csv'
    path = '../data/2018AIOpsData/kpi_normal_1.csv'
    # input_window = 50
    input_window = 100
    output_window = 1
    # scaler用于恢复原始数据
    train_data, val_data, _, scaler = get_data(path, input_window, output_window)
    train_data, val_data = train_data.to(device), val_data.to(device)
    batch_size = 32
    # 初始化模型
    feature = 512
    layers = 2
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    # 均方损失函数
    criterion = nn.MSELoss()
    # 学习率
    lr = 0.005
    # 定义优化器，SGD随机梯度下降优化
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # # 梯度下降优化算法：Adam自适应学习算法
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 每当scheduler.step()被调用step_size(=1)次，更新一次学习率，每次更新为当前学习率的0.95倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    # 记录指标信息
    writer = SummaryWriter(comment=f'{input_window}_{output_window}_{feature}_{layers}_{batch_size}_adam', flush_secs=10)
    epochs = 100
    best_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(train_data, batch_size, input_window)
        # 每10个epoch可视化一次测试集的loss
        if(epoch % 10 == 0):
            loss = plot_loss(model, val_data, batch_size, scaler, input_window)
        else:
            loss = evaluate(model, val_data, batch_size, input_window)
        print('-' * 75)
        print('|   End of epoch {:2d}   |   avg_loss {:5.5f}   |   time: {:5.2f}s   |'.format(epoch,
                                                                                              loss,
                                                                                              (time.time() - epoch_start_time)))
        # 存储最优模型
        if loss < best_loss:
            best_loss = loss
            best_model = model
        # 对lr进行调整（通常用在一个epoch中，放在train()之后的）
        scheduler.step()
    # 保存模型
    torch.save(best_model.state_dict(), f'./best_model/{input_window}_{output_window}_{feature}_{layers}_{batch_size}_adam.pth')
    print(f'total time: {time.time() - start_time},  best loss: {best_loss}')

    # 训练
    # for epoch in range(epoch_nums):
    #     model.train()
    #     for bach_idx, (features, targets) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         features, targets = features.view(-1,28*28).to(device), target.to(device)
    #         output = model(features)
    #         loss = criterion(output, targets)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    #         optimizer.step()
    #         total_loss += loss.cpu().item() * len(features)

    # 保存
    # state = {
    #     'epoch' : epoch + 1,                    # 当前的迭代次数
    #     'state_dict' : model.state_dict(),      # 模型参数
    #     'optimizer' : optimizer.state_dict()    # 优化器参数
    # }

    # 将state中的信息保存到checkpoint.pth.tar
    # torch.save(state, f'./checkpoint/checkpoint_{epoch}.pth.tar')     
    
    # Pytorch使用.tar格式来保存这些检查点
    # 恢复训练
    # checkpoint = torch.load(f'./checkpoint/checkpoint_{epoch}.pth.tar')
    # epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
