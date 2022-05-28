import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import read_csv
import sys
sys.path.append(os.getcwd())


torch.manual_seed(42)
np.random.seed(42)

# 输入窗口
input_window = 100
# 预测窗口
output_window = 1

batch_size = 20
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
            前向传播,将embedding后的输入加上position encoding\n
            x=(S,N,E),S是source sequence length, N是batch size,E是feature number\n
            即(batch_len,batch_size,d_model)

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """

        return x + self.pe[:x.size(0), :]


# 有学习参数的位置编码

# 位置编码的维数是可以优化的超参数
# class LearnedPositionEncoding(nn.Embedding):
#     def __init__(self,d_model, dropout = 0.1,max_len = 5000):
#         super().__init__(max_len, d_model)
#         self.dropout = nn.Dropout(p = dropout)

#     def forward(self, x):
#         weight = self.weight.data.unsqueeze(1)
#         x = x + weight[:x.size(0),:]
#         return self.dropout(x)


# torch.nn.Module是所有NN的基类

class TransformerModel(nn.Module):
    # 定义模型，继承nn.Module
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        """
        编码器Encoder，只有一层encoder层\n
        encoder层:10个头(默认8个)，dropout=0.1(默认),FNN默认维度2048，激活函数默认是ReLU\n
        CLASS torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')\n
        解码器Decoder，使用全连接层代替了Decoder， 可以加一下Transformer的Decoder试试效果

        Args:
            feature_size (int, optional): 向量维度，默认d_model=250
            num_layers (int, optional): encoder层数
            dropout (float, optional): 防止过拟合，默认0.1的概率随机丢弃
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        # torch.nn.Linear(in_features, out_features, bias=True)，输出维度是1
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange,initrange)

        # decoder：nn.Linear，设置bias和weight
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # 如果没有指定，就生成一个mask
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # 输入数据src在网络中进行前向传播
        # 首先添加位置编码，然后进过Encoder层，然后进入Decoder层，最后输出结果
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    #

    def _generate_square_subsequent_mask(self, sz):
        """
            为输入序列生成一个相同规模的square mask(方阵),在掩蔽的位置填充float('-inf')，正常位置填充float(0.0)\n
            首先生成上三角矩阵，然后转置mask，最后填充-inf达到掩蔽效果
        """
        # torch.ones(n, m)返回一个n*m的tensor
        # torch.triu(input, diagonal=0, out=None)→Tensor，input即生成的sz大小的tensor，diagonal为空保留输入矩阵主对角线与主对角线以上的元素，其他元素置0，然后将数字转换为True和False，然后转置mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # masked_fill(mask, value) → tensor，在mask值为1的位置处用value填充,mask的元素个数需要和tensor相同,但尺寸可以不同
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 输入input = seq->[0..99],输出target(label)->[1..100]


def create_inout_sequences(input_data, tw):
    """
        处理原始数据集得到模型的训练集，并转换为tensor\n
        train_sequence = create_inout_sequences(train_data,input_window)\n
        每次从数据集中取出目标窗口(tw)长度的数据：数据集的(i,i+100)部分是输入，数据集的(i+1,i+100+1)部分是输出，即监督学习的对应的输出(label)\n
        得到训练集:[[[0...100],[1...101]],[[1...101],[2...102]]...]

    Args:
        input_data: 原始训练集，即train_data
        tw: 输入窗口，即input_window = 100
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    # 将训练集转换成tensor
    return torch.FloatTensor(inout_seq)


def get_data():
    """
        导入CSV数据，对数据做归一化处理,提升模型的收敛速度,提升模型的精度\n
        初始化scaler在(-1,1)之间,然后使用scaler归一化数据，amplitude指序列振幅\n
        根据sampels将数据划分为数据集和测试集，调用create_inout_sequences()将其转换为tensor
    """

    # header=0，使用数据文件的第一行作为列名称，将第一列作为索引
    series = read_csv('./data/minutedata.csv', header=0,
                      index_col=0, parse_dates=True, squeeze=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # reshape()更改数据的行列数，(-1, 1)将series变为一列 (2203,1)，归一化后再(-1)变为一行 (2203,)
    amplitude = scaler.fit_transform(
        series.to_numpy().reshape(-1, 1)).reshape(-1)
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
        把源数据细分为长度为batch_size的块，生成模型训练的输入序列和目标序列
        data, targets = get_batch(train_data, i, batch_size)
    Args:
        source: 即train_data
        i: 每组数据从i开始,即当前batch的起始索引
        batch_size: 10
    """
    seq_len = min(batch_size, len(source) - 1 - i)
    # 每个batch的数据
    data = source[i:i+seq_len]
    # torch.stack(inputs, dim=?)→Tensor，对inputs(多个tensor)沿指定维度dim拼接，返回一维的tensor
    # 即source = [([0...100],[1...101]),([1...101],[2...102])...]，取出item[0]拼接,即train_seq = [[0...100],[1...101]...]
    # torch.chunk(tensor, chunk_num, dim)将tensor在指定维度上(0行,1列)分为n块,返回一个tensor list,是一个tuple
    # 即将拼接后的source按每一列分成一个tensor
    # item[0]是输出,item[1]是对应的target
    input = torch.stack(torch.stack([item[0]
                        for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1]
                         for item in data]).chunk(input_window, 1))
    return input, target

# train_data = create_inout_sequences(train_data,input_window)的[:-output_window]部分
# 训练集:[[[0,100],[1,101]],[[1,101],[2,102]]...]


def train(train_data):
    # 设置为trainning模式,启用BatchNormalization和Dropout
    model.train()
    # 一个epoch总的损失
    total_loss = 0.
    start_time = time.time()

    # 将训练集按batch_size=10划分成一个个batch训练,所有的batch计算为一次epoch
    # range()按batch_size步长生成索引, enumerate()同时列出数据和数据索引
    # 根据每次划分得到的i获取每一个batch的数据

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
#         反向传播前将梯度清零，即将loss关于weight的导数变成0
        optimizer.zero_grad()
#         前向传播,即把数据输入网络中并得到输出
        output = model(data)
#         均方损失函数:criterion = nn.MSELoss() = (x-y)^2
        loss = criterion(output, targets)
#         反向传播梯度
        loss.backward()
#         梯度裁剪:在BP过程中会产生梯度消失（偏导无限接近0）解决方法是设定一个阈值,当梯度小于阈值时更新的梯度为阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
#         根据梯度更新网络参数
        optimizer.step()

        # 获取loss的标量item()得到一个元素张量里面的元素值，即将一个零维张量转换成浮点数
        total_loss += loss.item()
#         打印训练信息
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
#             训练时间
            elapsed = time.time() - start_time
#             打印日志:第几个epoch,第几个batch,一个epoch的batch总数,学习率,损失函数,训练时间
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                      epoch, batch, len(
                          train_data) // batch_size, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# 可视化损失函数

def plot_and_loss(eval_model, data_source, epoch):

    #     设置为evaluation模式,不启用BatchNormalization和Dropout,将BatchNormalization和Dropout置为False
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat(
                (test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    len(test_result)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.plot(test_result-truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.close()

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

    # I used this plot to visualize if the model pics up any long therm struccture within the data.
    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png' % steps)
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
    # 获取训练数据集和测试数据集
    train_data, val_data = get_data()
    # 初始化模型
    model = TransformerModel().to(device)
    # 定义均方损失函数
    criterion = nn.MSELoss()
    # 学习率
    lr = 0.005
    # 随机梯度下降优化
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 梯度下降优化算法：Adam自适应学习算法
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    # 每step_size个epoch后更新一次学习率，每次更新为当前学习率的0.95倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # 设置100个epochs
    epochs = 100

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # 开始训练
        train(train_data)
        # 每10个epoch打印一次信息
        if(epoch % 10 is 0):
            val_loss = plot_and_loss(model, val_data, epoch)
            predict_future(model, val_data, 200)
        else:
            val_loss = evaluate(model, val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                                                      val_loss, math.exp(val_loss)))
        print('-' * 89)

        # 存储最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
