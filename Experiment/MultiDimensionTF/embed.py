import math
import torch
import torch.nn as nn


# 无学习参数的位置编码
class PositionalEncoding(nn.Module):

    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TokenEmbedding(nn.Module):
    """
    对输入的原始数据(7列)进行一个1维卷积,得到一个d_model=512维的向量
    看成7个矩形堆叠，每一个矩形，长是数据行数，宽是指标个（列）数
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # in_channels(int): 输入的通道(文本分类中即为词向量的维度), out_channels(int): 卷积产生的通道, 有多少个out_channels, 就需要多少个1维卷积核
        # kernel_size(int or tuple): 卷积核的大小为(k,?), 第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        # padding (int or tuple, optional)：输入的每一条边补充0的层数
        # padding_mode：填充模式，circula即循环填充
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # kaiming正态分布，fan_in使正向传播时，方差一致
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # permute(0, 2, 1)，交换第1维和第2维
        # transpose(1,2)，转置第1维和第2维
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x