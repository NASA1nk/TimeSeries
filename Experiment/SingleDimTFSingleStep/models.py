import torch
import torch.nn as nn


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
