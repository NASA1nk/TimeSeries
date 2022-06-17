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
from embed import PositionalEncoding
from models import TransformerModel
# 测训练过程可视化
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0,os.getcwd())

torch.manual_seed(42)
np.random.seed(42)

input_window = 100
output_window = 1

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 位置编码,得到token的绝对位置信息和相对位置信息
# 构造一个跟输入embedding维度一样的矩阵,然后跟输入embedding相加得到multi-head attention的输入

# 无学习参数的位置编码
class PositionalEncoding(nn.Module):

    # d_model表示向量维度，max_len表示最大长度为5000，一般是200
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # zeros()生成二维矩阵pe(5000*250)，len行，d_model列，即5000个维度为250的行向量
        pe = torch.zeros(max_len, d_model)

        # arrange()生成一维矩阵pos(5000)，即5000个数(维度为5000的一个行向量)
        # unsqueeze()用于在指定位置增加维度，如(0，1，2)三维，返回的tensor与输入的tensor共享内存，即改变其中一个的内容也会改变另一个
        # 1即表示在第二个维度处增加，即列方向上增加一个维度，维度为5000的一个行向量变成了5000个维度为1的行向量，pos变成了二维矩阵(5000*1)
        # pos这5000个一维向量就构成一列，计算位置信息来依次填充pe的每一个奇偶列
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 2是arange()中的步长参数，位置编码计算公式 e^(2i*-log10000/d_model)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 填充pe矩阵，偶数列正弦编码，奇数列余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 0表示在第一个维度处增加一个维度，则二维矩阵pe(5000*250)变成了三维矩阵(1*5000*250)
        # 第一个维度用于接受batch_size参数，表示第几个batch
        # transpose(0,1)用于对矩阵进行转置，转置0维和1维，将(1*5000*250)变成(5000*1*250)
        # 即位置编码三维矩阵pe最终是5000个，每个250维的行向量组成
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False

        # pytorch一般情况下将网络中的参数保存成OrderedDict形式
        # 网络参数包括2种，一种是模型中各种module含的参数，即nn.Parameter，也可以在网络中定义其他的nn.Parameter参数，另外一种是buffer
        # nn.Parameter会在每次optim.step会得到更新（即梯度更新），buffer则不会被更新（不会有梯度传播给它），但是能被模型的state_dict记录下来，buffer的更新在forward中

        # 将pe存到内存中的一个常量(映射)，模型保存和加载的时候可以写入和读出，可以在forward()中使用
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            前向传播,将embedding后的输入加上position encoding
            x=(S,N,E),S是source sequence length, N是batch size,E是feature number
            即(source_sequence_length, batch_size, d_model)
        """

        return x + self.pe[:x.size(0), :]


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


def get_data(path):
    """
        导入CSV数据，对数据做归一化处理,提升模型的收敛速度,提升模型的精度
        初始化scaler在(-1,1)之间,然后使用scaler归一化数据，amplitude指序列振幅
        根据sample将数据划分为数据集和测试集，调用create_inout_sequences()将其转换为tensor
    """

    # header=0，使用数据文件的第一行作为列名称，将第一列作为索引
    df = pd.read_csv(path, header=0, index_col=0, parse_dates=True, squeeze=True)
    # series = df.to_numpy()
    # 通过df.loc来限制行列
    series = df['value'].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # reshape()更改数据的行列数，(-1, 1)将df变为一列 (2203,1)，归一化后再(-1)变为一行 (2203,)
    # amplitude = scaler.fit_transform(df.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)
    # 反归一化：reamplitude = scaler.inverse_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # # 多维数据
    # data = df.loc['col1','col2']
    # series = df.to_numpy()
    # amplitude = scaler.fit_transform(series)

    sample = 100000
    train_data = amplitude[:sample]
    test_data = amplitude[sample:]

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
            # 以batch为单位计算的loss,所以乘以batch size(最后一次不足一个batch,所以是data[0]),最后计算平均
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
            data, targets = get_batch(data_source, i, 1)
            output = eval_model(data)
            # 数据长度为1
            loss = criterion(output, targets).cpu().item()
            total_loss += loss
            # torch.cat：将两个tensor拼接在一起，cat即concatenate
            # 拼接维数dim可以不同，其余维数要相同才能对其，二维的0表示按行（维数0）拼接，1表示按列（维数1)拼接
            # 这里是拼接每个batch中，网络的输出列表和当前标签label列表的最后一个值
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, targets[-1].view(-1).cpu()), 0)
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
    # batch_size = 1, 即(100,1,1)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        # 在data上以input_window大小的滑动窗口来一步步预测
        for i in range(0, steps):
            # 在第一次预测的时候,data[-input_window:]就是data,因为data是根据input_window划分来的
            output = eval_model(data[-input_window:])
            # 拼接n步的结果,output[-1:]即(1,1,1)
            data = torch.cat((data, output[-1:]))
    # 最后拼接的data即(100,1,100+steps)
    data = data.cpu().view(-1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(data, c='red', linestyle='-.', label='predict')
    ax.plot(data[:input_window], c='blue', label='ground_truth')
    ax.legend()
    plt.savefig(f'./Experiment/TransformerSingleStep/img/predict_{steps}_{epoch/10}.png')


if __name__ == "__main__":
    # 目前512_1_32_0.005loss最小
    s_time = time.time()
    writer = SummaryWriter(comment='256_1_64',flush_secs=60)
    torch.cuda.set_device(0)
    # 指定device，后续可以调用to(device)把Tensor移动到device上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输入窗口
    input_window = 100
    # 预测窗口
    output_window = 1
    batch_size = 64
    epochs = 100
    path = './Experiment/data/2018AIOpsData/kpi_1.csv'
    # 获取训练数据集和测试数据集
    train_data, val_data = get_data(path)
    # 初始化模型（实例化网络），然后迁移到gpu上
    model = TransformerModel().to(device)
    # 均方损失函数：nn.MSELoss() = (x-y)^2/n，逐元素运算
    criterion = nn.MSELoss()
    # 学习率
    # lr = 0.005
    lr = 0.01
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
    e_time = time.time()
    print(f'total time:{e_time - s_time}')
    