import math
import torch
import torch.nn as nn


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
