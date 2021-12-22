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

二维输入：矩阵

- 每一行：样本x
- 每一列：特征f

> 一个batch就是一个二维矩阵（一个正方形）

**Batch Norm**

Batch Norm是**每一次将一列向量（一个特征）在一个batch中，均值变为0，方差变为1**

- 将向量本身的均值减去向量的均值，再除以向量的方差 
- 训练的时候是在一个小batch中计算
- **预测的时候需要计算一个全局的均值和方差使用**
  - 因为是所有样本的特征来计算

> Batch Norm还会学习两个参数，λ和β。将一个向量表示为任意的均值和方差

**Layer Norm**

Layer Norm是**每一次将一行向量（一个样本）在一个batch中，均值变为0，方差变为1**

- 可以认为是将向量转置后使用Batch Norm然后变换后再转置回来就得到结果
- **不需要全局的均值和方差，因为计算是针对每一个样本的**

但在**Transformer中，输入的一个序列样本是三维的**

- 每一个序列样本里面有很多元素，每个元素对应一个长度的向量

三维的信息

- 长：一个长度为n的seq
- 宽：元素对应的一个长为d=512的向量
  - 每一个维度就是一个特征
- 高：batch_size

如果选取Batch Norm，每一次取一个特征，即所有序列的所有元素对应向量的一个维度

- 就是对应**竖着切**正方体的一个切面：一个正方形
- **将这个面拉直成一条直线**，将其均值变为0，方差变为1

如果选取Layer Norm，每一次取一个样本，即一个序列的所有元素对应向量的所有维度

- 就是对应**横着切**正方体的一个切面：一个正方形
- **将这个面拉直成一条直线**，将其均值变为0，方差变为1

> 不同的切法带来不同的结果

**为什么在变长的序列中不使用batch norm**

因为时序的模型中，每个样本的长度可能不一样（补0填充），所以不同的切法得到的结果不一样，计算均值和方差也不一样

- 所以如果不同的序列长度变化比较大，batch norm每次计算均值和方差抖动就比较大

- 做预测时要计算全局的均值和方差，如果新预测的样本长度更大，与之前训练的序列长度差距较大，那么之前算的均值和方差的效果可能就不好
- 而Layer Norm是在给每个样本计算均值和方差，没有太大影响

## Decoder

6 layers：3sub-layers

- multi-attention
- MLP：positionwise fully connected feed-forward network
- **masked multi-attention**

**使用自回归来做预测时，当前的输入集是上一个时刻的输出。不应该看到之后那些时刻的输出，但是注意力机制每次都是看到整个完整的输入**

- 所以通过一个**带掩码的注意力机制**，来避免**训练的时候**解码器预测第t个时刻的输出的时候看到t时刻后面的输出
- 从而保证训练和预测的行为是一致的  

## Attention

注意力层

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力函数是将一个query向量和一些key-value向量对映射成一个输出的函数

- 输出就是value向量的一个加权和，所以输出的维度和value的维度相同
- 对每一个value向量的权重，是value向量对应的key向量和query向量相似度计算得出
  - **相似度对于不同的注意力机制有不同的计算结果**
  - 所以相同的key-value，对不同的query会得出不同的结果

### Scaled Dot-Product Attention

Transformer使用的是Scaled Dot-Product Attention（最简单的注意力机制），使用内积做计算，**内积值越大，相似度越高**

- query向量和key向量都是等长的，等于dk
- value向量的长度是dv，即输出长度也是dv

Layer Norm后向量的维度等会$\sqrt{d_k}$，所以除以$\sqrt{d_k}$



