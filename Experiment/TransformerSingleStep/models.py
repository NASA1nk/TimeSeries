import os
import sys
import time
import math
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter
from get_data import get_data, get_batch
from embed import PositionalEncoding


sys.path.insert(0,os.getcwd())


torch.manual_seed(42)
np.random.seed(42)


# torch.nn.Module是所有NN的基类
class TransformerModel(nn.Module):
    # 定义模型网络结构
    def __init__(self, feature_size=256, num_layers=1, dropout=0.1):
        """
        编码器Encoder，只有一层encoder层
        encoder层:10个头(默认8个)，dropout=0.1(默认),FNN默认维度2048，激活函数默认是ReLU
        torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
        解码器Decoder，使用全连接层代替了Decoder， 可以用Transformer的Decoder试试效果

        Args:
            feature_size (int, optional): 向量维度，默认d_model=250
            num_layers (int, optional): encoder层数
            dropout (float, optional): 防止过拟合，默认0.1的概率随机丢弃
        """
        # 获取父类nn.Module，然后调用父类的构造函数，初始化一些必要的变量和参数
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
                                                        d_model=feature_size, 
                                                        nhead=8, 
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        # torch.nn.Linear(in_features, out_features, bias=True)，输出维度是1
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange,initrange)

        # decoder：nn.Linear，设置bias和weight（pytorch特性）
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # 模型参数的前向传播
    def forward(self, src):
        # 如果没有指定，就生成一个mask
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # 输入数据src在网络中进行前向传播
        # 首先添加位置编码，然后进过Encoder层，然后进入Decoder层，最后输出结果
        src = self.pos_encoder(src)
        
        # 在这里添加时间编码

        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


    def _generate_square_subsequent_mask(self, len):
        """
            调用：_generate_square_subsequent_mask(len(src))
            为输入序列生成一个相同规模的square mask(方阵),在掩蔽的位置填充float('-inf')，正常位置填充float(0.0)
            首先生成上三角矩阵，然后转置mask，最后填充-inf达到掩蔽效果
        """
        # torch.ones(n, m)返回一个n*m的tensor，这里根据len(src) = 100
        # torch.triu(input, diagonal=0, out=None)→Tensor，input即生成的len=100大小的tensor，diagonal为空保留输入矩阵主对角线与主对角线以上的元素，其他元素置0（即上三角矩阵），然后将数字转换为True和False，然后转置mask
        mask = (torch.triu(torch.ones(len, len)) == 1).transpose(0, 1)
        # masked_fill(mask, value) → tensor，在mask值为1的位置处用value填充,mask的元素个数需要和tensor相同,但尺寸可以不同
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# train_data = create_inout_sequences(train_data,input_window)的[:-output_window]部分
# 训练集:[[[0,100],[1,101]],[[1,101],[2,102]]...]
def train(train_data):
    # 设置模型为trainning模式,启用BatchNormalization和Dropout
    model.train()
    # 一个epoch总的损失
    total_loss = 0.
    avg_loss = 0.
    start_time = time.time()
    # 根据每次划分得到的i获取每一个batch的数据
    for batch, i in enumerate(range(0, len(train_data)-1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
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
            print('| epoch {:2d} | {:5d} / {:} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:5.2f}'.format(
                                                    epoch, 
                                                    batch, len(train_data) // batch_size,
                                                    scheduler.get_last_lr()[0],
                                                    cost_time / log_interval * 1000,
                                                    cur_loss, 
                                                    math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    avg_loss = avg_loss / len(train_data)
    writer.add_scalar('train_loss', avg_loss, epoch) 
    

# 评估训练后的模型
def evaluate(eval_model, data_source):
    # 设置为evaluation模式，不启用BatchNormalization和Dropout，将BatchNormalization和Dropout置为False（training的属性置为False）
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 64
    # 表明当前计算不需要反向传播
    with torch.no_grad():
        for i in range(0, len(data_source)-1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            # 模型的评估和模型的训练逻辑基本相同，唯一的区别是评估只需要forward pass，不需要backward pass
            output = eval_model(data)
            # 没有反向传播，可以直接获取loss的标量item()
            loss = criterion(output, targets).cpu().item()
            # 以batch为单位计算的loss
            total_loss += len(data[0]) * loss
    # 返回整个验证集的所有元素的平均MSEloss
    avg_loss = total_loss / len(data_source)
    # 记录关键指标,保存在本地
    writer.add_scalar('eval_loss', avg_loss, epoch)
    return avg_loss


# 可视化训练损失
def plot_and_loss(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        # 相当于batch size = 1
        for i in range(0, len(data_source) - 1):
            # 传入1
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            # 数据长度为1
            total_loss += criterion(output, target).item()
            # torch.cat：将两个tensor拼接在一起，cat即concatenate
            # 拼接维数dim可以不同，其余维数要相同才能对其，二维的0表示按行（维数0）拼接，1表示按列（维数1)拼接
            # 这里是拼接每个batch中，网络的输出列表和当前标签label列表的最后一个值
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
    # 画图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    # 红色是模型生成的预测值，蓝色是label，绿色是差值diff，即每个batch的
    ax.plot(test_result, c='red', label='predict')
    ax.plot(truth, c='blue', label='ground_truth')
    ax.plot(test_result-truth, color="green", label="diff")
    ax.legend() 
    plt.savefig(f'./Experiment/TransformerSingleStep/img/Epoch-{epoch}.png')
    # 返回验证集所有数据的平均MSEloss
    return total_loss / i


# predict the next n steps based on the input data
def predict(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    # batch_size = 1
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            # 拼接n步的结果
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(data, c='red', linestyle='-.', label='predict')
    ax.plot(data[:input_window], c='blue', label='ground_truth')
    ax.legend()
    plt.savefig(f'./Experiment/TransformerSingleStep/img/predict_{steps}_{epoch/10}.png')


if __name__ == "__main__":
    writer = SummaryWriter(comment='512_1_64',flush_secs=60)
    torch.cuda.set_device(1)
    # 指定device，后续可以调用to(device)把Tensor移动到device上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输入窗口
    input_window = 100
    # 预测窗口
    output_window = 1
    batch_size = 64
    epochs = 100

    path = './Experiment/data/2018AIOpsData/KPIData/kpi_1.csv'
    # 获取训练数据集和测试数据集
    train_data, val_data = get_data(path)
    # 初始化模型（实例化网络），然后迁移到gpu上
    model = TransformerModel(feature_size=512).to(device)
    # 均方损失函数：nn.MSELoss() = (x-y)^2/n，逐元素运算
    criterion = nn.MSELoss()
    # 学习率
    lr = 0.005
    # 定义优化器，SGD随机梯度下降优化
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # # 梯度下降优化算法：Adam自适应学习算法
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    # step_size参数表示每当scheduler.step()被调用step_size次，更新一次学习率，每次更新为当前学习率的0.95倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        # 开始训练
        train(train_data)
        # 每10个epoch打印一次信息，并且预测一次
        if(epoch % 10 is 0):
            val_loss = plot_and_loss(model, val_data)
            # 预测20步
            predict(model, val_data, 10)
        else:
            val_loss = evaluate(model, val_data)
        print('-' * 85)
        print('| End of epoch {:2d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:4.2f}'.format(
                                                                                                    epoch, 
                                                                                                    (time.time() - epoch_start_time),
                                                                                                    val_loss, 
                                                                                                    math.exp(val_loss)))
        print('-' * 85)
        # 存储最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            if(epoch % 20 is 0):
                torch.save(best_model.state_dict(), f'./Experiment/TransformerSingleStep/save_model/model_epoch_{epoch}.pth')   
        # 对lr进行调整（通常用在一个epoch中，放在train()之后的）
        scheduler.step()
    torch.save(best_model.state_dict(), f'./Experiment/TransformerSingleStep/save_model/best_model.pth') 

    # # 恢复模型
    # new_model = model = TransformerModel()        
    # # 将model中的参数加载到new_model中            
    # new_model.load_state_dict(torch.load(PATH))   


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
    #         if not batch_idx % 50:
    #             print ('Epoch: %03d/%03d | Batch %03d/%03d | loss: %.4f' %(
    #                                                                     epoch+1, 
    #                                                                     epoch_nums, 
    #                                                                     batch_idx, 
    #                                                                     len(train_loader), 
    #                                                                     loss.item()))

    # 测试
    # model.eval()
    # preds = []
    # # test dataset
    # for x in test_set:
    #     x = x.to(device)
    #     with torch.no_grad():
    #         pred = model(x)
    #         # collect predictio
    #         preds.append(pred.cpu())

    # # 保存
    # state = {
    #     'epoch' : epoch + 1,                    # 当前的迭代次数
    #     'state_dict' : model.state_dict(),      # 模型参数
    #     'optimizer' : optimizer.state_dict()    # 优化器参数
    # }
    # torch.save(state, f'./checkpoint/checkpoint_{epoch}.pth.tar')     #将state中的信息保存到checkpoint.pth.tar
    # #Pytorch 约定使用.tar格式来保存这些检查点
    # # 恢复训练
    # checkpoint = torch.load(f'./checkpoint/checkpoint_{epoch}.pth.tar')
    # epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
