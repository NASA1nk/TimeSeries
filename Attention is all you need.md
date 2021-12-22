# 引言

传统的seq2seq依赖使用encoder-decoder结构的RNN,CNN，在encoder-decoder之间会使用注意力机制

- **RNN**：对第t个向量，计算它的隐藏状态ht，ht是由ht-1和当前向量决定，这样可以将前面学到的信息通过隐藏状态ht-1传递到当前

  - 从左往右一步步走，时序过程，难以并行

  - 如果时序比较长，早期的信息在后面可能会丢失，如果不想丢失，就得设置比较大的ht，但每一个时间步都要存储ht，比较占内存

- **Attention**：attention之前就被用在encoder-decoder结构中
  - 主要关注于怎么把encoder的输出有效的传递到decoder中

**Transformer只是一个更简单的架构（simple） ，不使用循环和卷积，只使用了纯注意力机制**

- 将recurrent layers替换成multi-headed self-attention
- 更快，完全并行

> 一开始的Transformer只针对机器翻译领域

# 背景

## CNN

使用CNN替换RNN减少时序的计算，但CNN对比较长的序列难以建模

- 使用卷积核每次看一个比较小的窗口
- 如果两个像素隔得比较远，要卷很多层

CNN的优点是可以有多个输出通道，每个输出可以看成识别的不一样的模式

所以提出multi-headed self-attention，用多头来模拟CNN的多输出通道

## memory networks

# 方法

## encoder-decoder

编码器-解码器架构

- encoder将原始的输入表示为机器学习可以理解的一组向量
  - 将input序列x=(x1,x2,x3,x4,...,xn)映射成一组向量序列z=(z1,z2,z3,z4,...,zn)
- decoder根据向量序列z=(z1,z2,z3,z4,...,zn)生成y=(y1,y2,y3,y4,...,ym)
  - 自回归一个个生成yi
  - 过去时刻的输出也会当成当前时刻的输入 

## Encoder

6 layers：2sub-layers

- multi-attention
- MLP：positionwise fully connected feed-forward network

`LayerNorm(x+Sublayer(x))`

因为残差连接需要一样的大小，否则要做投影，所以规定每一个layer的输出维度（dimension）dmodel = 512

所以超参数只有2个

- 多少层layer
- 多少个维度dmodel

## Layer Norm

为什么在变长的序列中不使用batch norm