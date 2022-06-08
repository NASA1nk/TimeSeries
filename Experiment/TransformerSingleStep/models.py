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
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter
from get_data import get_data, get_batch
from embed import PositionalEncoding


sys.path.insert(0,os.getcwd())

writer = SummaryWriter('./logs')
torch.manual_seed(42)
np.random.seed(42)

# 输入窗口
input_window = 100
# 预测窗口
output_window = 1

# 指定device，后续可以调用to(device)把Tensor移动到device上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.nn.Module是所有NN的基类

class TransformerModel(nn.Module):
    # 定义模型网络结构
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
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
                                                        nhead=10, 
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
    start_time = time.time()
    # 根据每次划分得到的i获取每一个batch的数据
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        # 反向传播前将梯度清零，即将loss关于weight的导数变成0
        optimizer.zero_grad()
        # 前向传播,即把数据输入网络（调模型中的forward函数）中并得到输出
        output = model(data)
        # 均方损失函数:criterion = nn.MSELoss() = (x-y)^2
        loss = criterion(output, targets)
        # 反向传播梯度
        loss.backward()
        # 梯度裁剪:在BP过程中会产生梯度消失（偏导无限接近0）解决方法是设定一个阈值,当梯度小于阈值时更新的梯度为阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        # 根据梯度反向传播来更新网络参数（通常用在每个batch中，应该在train()中，只有这样模型才会更新）
        optimizer.step()
        # 获取loss的标量item()得到一个元素张量里面的元素值，即将一个零维张量转换成浮点数
        total_loss += loss.item()

        # 记录关键指标,保存在本地
        writer.add_scalar('./train/loss', loss.item(), global_step=batch)
        # writer.add_scalar('loss', loss, global_step=epoch)
        # 打印训练信息
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            # 训练时间
            elapsed = time.time() - start_time
            # 打印日志:第几个epoch,第几个batch,一个epoch的batch总数,学习率,损失函数,训练时间
            print('| epoch {:2d} | {:5d} / {:} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:5.2f}'.format(
                                                    epoch, 
                                                    batch, 
                                                    len(train_data) // batch_size,
                                                    scheduler.get_last_lr()[0],
                                                    elapsed*1000 / log_interval,
                                                    cur_loss, 
                                                    math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# 可视化损失函数
def plot_and_loss(eval_model, data_source, epoch):
    """
        调用：val_loss = plot_and_loss(model, val_data, epoch)，传入的是验证集val_data
        
    """
    # 设置为evaluation模式，不启用BatchNormalization和Dropout，将BatchNormalization和Dropout置为False（training的属性置为False）
    eval_model.eval()
    # 初始化
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    # 表明当前计算不需要反向传播
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            # 得到模型的结果
            output = eval_model(data)
            # 因为没有反向传播，可以直接获取loss的标量item()
            total_loss += criterion(output, target).item()
            writer.add_scalar('./eval/loss', criterion(output, target).item(), global_step=i)
            # torch.cat：将两个tensor拼接在一起，cat即concatenate
            # 拼接维数dim可以不同，其余维数要相同才能对其，二维的0表示按行（维数0）拼接，1表示按列（维数1)拼接
            # 这里是拼接每个batch中，网络的输出列表和当前标签label列表的最后一个值
            test_result = torch.cat(
                (test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # 红色是模型生成的预测值，蓝色是label，绿色是差值，即每个batch的loss
    pyplot.plot(test_result, color="red")
    # pyplot.plot(truth[:500], color="blue")
    pyplot.plot(truth, color="blue")
    pyplot.plot(test_result-truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.savefig('./img/SingleStep-Epoch%d.png' % epoch)
    pyplot.savefig('./Experiment/TransformerSingleStep/img/Epoch-%d.png' % epoch)
    pyplot.close()

    # 返回验证集的一个epoch的平均loss
    return total_loss / i

# predict the next n steps based on the input data
def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('./Experiment/TransformerSingleStep/img/SingleStep-Epoch%d.png' % steps)
    pyplot.close()


# 模型的评估和模型的训练逻辑基本相同，唯一的区别是评估只需要forward pass，不需要backward pass
def evaluate(eval_model, data_source):
    # Turn on the evaluation mode
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * \
                criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


if __name__ == "__main__":
    path = './Experiment/data/2018AIOpsData/KPIData/kpi_1.csv'
    # 获取训练数据集和测试数据集
    train_data, val_data = get_data(path)
    # 初始化模型（实例化网络），然后迁移到gpu上
    model = TransformerModel().to(device)
    # 定义均方损失函数
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
    # 设置100个epochs
    batch_size = 32
    epochs = 100
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        # 开始训练
        train(train_data)
        # 每10个epoch打印一次信息，并且预测一次
        if(epoch % 10 is 0):
            val_loss = plot_and_loss(model, val_data, epoch)
            predict_future(model, val_data, 100)
        else:
            val_loss = evaluate(model, val_data)

        print('-' * 89)
        print('| End of epoch {:2d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:4.2f}'.format(
                                                                                                    epoch, 
                                                                                                    (time.time() - epoch_start_time),
                                                                                                    val_loss, math.exp(val_loss)))
        print('-' * 89)
        # 存储最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
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
    #         features = features.view(-1,28*28).to(device)
    #         targets = target.to(device)
    #         output = model(features)
    #         loss = criterion(output, targets)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    #         optimizer.step()
    #         if not batch_idx % 50:
    #             print ('Epoch: %03d/%03d | Batch %03d/%03d | loss: %.4f' %(
    #                                                                     epoch+1, 
    #                                                                     epoch_nums, 
    #                                                                     batch_idx, 
    #                                                                     len(train_loader), 
    #                                                                     loss.item()))
