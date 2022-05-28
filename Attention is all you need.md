# 李沐-Transformer论文逐段精读

一开始的Transformer只针对机器翻译领域

## 引言

传统的seq2seq依赖使用**encoder-decoder结构**的RNN，CNN，在encoder-decoder之间会使用注意力机制

- **RNN**：对第t个向量，计算它的隐藏状态ht，**ht是由ht-1和当前向量决定**，这样可以将前面学到的信息通过隐藏状态ht-1传递到当前
  - 从左往右一步步走，时序过程，难以并行，模型效率十分低
  - 如果时序比较长，早期的信息在后面可能会丢失，如果不想丢失，就得设置比较大的ht，但每一个时间步都要存储ht，比较占内存
- **Attention**：attention之前就被用在encoder-decoder结构中
  - 主要关注于怎么把encoder的输出有效的传递到decoder中

**Transformer只是一个更简单的架构（simple） ，不使用循环和卷积，只使用了纯注意力机制**

- 将recurrent layers替换成multi-headed self-attention
- 更快，完全并行

## 背景

### CNN

使用CNN替换RNN减少时序的计算，**但CNN对比较长的序列难以建模**

- 使用卷积核每次看一个比较小的窗口
- 如果两个像素隔得比较远，要卷很多层

**CNN的优点是可以有多个输出通道，每个输出可以看成识别的不一样的模式**，所以提出multi-headed self-attention，**用多头来模拟CNN的多输出通道**

- 多头也可以识别不同的模式

### memory networks

## 方法

### Encoder-Decoder

**编码器-解码器架构**

- **Encoder将原始的输入表示为机器学习可以理解的一组向量**
  - 将input序列x=(x1,x2,x3,x4,...,xn)映射成一组向量序列z=(z1,z2,z3,z4,...,zn)
- Decoder根据向量序列z=(z1,z2,z3,z4,...,zn)生成y=(y1,y2,y3,y4,...,ym)
  - **自回归一个个生成yi**
  - 过去时刻的输出也会当成当前时刻的输入 

![Transformer架构](Attention is all you need.assets/Transformer架构.png)

### Encoder

6 layers：2sub-layers

- Multi-Head Attention
- MLP：positionwise fully connected feed-forward network

`LayerNorm(x + Sublayer(x))`

- **因为残差连接需要一样的大小，否则要做投影**，所以规定每一个layer的输出维度（dimension）dmodel = 512

所以超参数只有2个

- 多少层layer
- 多少个维度dmodel

### Layer Norm

二维输入：矩阵

- 每一行：样本x
- 每一列：特征f

> 一个batch就是一个二维矩阵（一个正方形）

#### Batch Norm

Batch Norm是**每一次将一列向量（一个特征）在一个batch中，均值变为0，方差变为1**

- 将向量本身的均值减去向量的均值，再除以向量的方差 
- 训练的时候是在一个小batch中计算
- **预测的时候需要计算一个全局的均值和方差使用**
  - 因为是所有样本的特征来计算

> Batch Norm还会学习两个参数，λ和β，用于将一个向量表示为任意的均值和方差

#### Layer Norm

Layer Norm是**每一次将一行向量（一个样本）在一个batch中，均值变为0，方差变为1**

- **不需要全局的均值和方差，因为计算是针对每一个样本的**

> 可以认为是**将向量转置后使用Batch Norm然后变换后再转置回来就得到结果**

#### 三维数据

但在**Transformer中，输入的一个序列样本是三维的**

> 每一个序列样本里面有很多元素，每个元素对应一个长度的向量

三维数据的信息

- 长：一个长度为n的seq，有n个元素
- 宽：一个元素对应为一个长为dmodel=512的向量（每一个维度就是一个特征）
- **高：batch size**

#### Norm区别

不同的切法带来不同的结果

- 如果选取Batch Norm，每一次取一个特征，即**所有序列的所有元素的对应向量的一个维度**

  - 就是对应**竖着切**正方体的一个切面：一个正方形

  - **将这个面拉直成一条直线，将其均值变为0，方差变为1**


- 如果选取Layer Norm，每一次取一个样本，即**一个序列的所有元素的对应向量的所有维度**

  - 就是对应**横着切**正方体的一个切面：一个正方形

  - **将这个面拉直成一条直线**，将其均值变为0，方差变为1


**为什么在变长的序列中不使用batch norm**

因为时序的模型中，**每个样本的长度可能不一样（补0填充）**，所以不同的切法得到的结果不一样，计算均值和方差也不一样

- 所以如果不同的序列长度变化比较大，batch norm每次计算均值和方差抖动就比较大

- 做预测时要计算全局的均值和方差，如果新预测的样本长度更大，与之前训练的序列长度差距较大，那么之前算的均值和方差的效果可能就不好
- 而Layer Norm是在给每个样本计算均值和方差，没有太大影响

### Decoder

6 layers：3sub-layers

- Multi-Head Attention
- MLP：positionwise fully connected feed-forward network
- **masked multi-attention**

> 掩码多头

**使用自回归来做预测时，当前的输入集是上一个时刻的输出，当前的输出是下一个时刻的输入，但是不应该看到之后那些时刻的输入（会影响当前的输出），但是注意力机制每次都是看到整个完整的输入**

- 所以通过一个带掩码的注意力机制，来避免训练的时候解码器预测第t个时刻的输出的时候看到t时刻后面的输入

- **从而保证训练和预测的行为是一致的**  

### Attention

注意力层

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

**注意力函数是将一个query向量和一些key-value向量对映射成一个输出的函数**

- **输出就是value向量的一个加权和**，所以输出的维度和value的维度相同
- **每一个value向量的权重是value向量对应的key向量和query向量相似度计算得出**
  - **所以相同的key-value，对不同的query会得出不同的结果**
  - 相似度对于不同的注意力机制有不同的计算结果

#### Scaled Dot-Product Attention

Transformer使用的是**Scaled** Dot-Product Attention（最简单的注意力机制），使用内积做计算，**内积值越大，相似度越高**

$Q_{n×d_k}$，$K_{m×d_k}$，$Q×K = A_{n×m}$，$V_{m×d_v}$，$A×V = A'_{n×d_v}$

- query向量和key-value向量对的个数可能是不一样的（key-value向量对，所以key和value的个数相同）
- query向量和key向量长度都是相同的，等于dk
- value向量的长度是dv，**即输出长度也是dv**

**缩放因子**

Layer Norm后向量的维度等会$\sqrt{d_k}$，所以除以$\sqrt{d_k}$

- 当$d_k$不是很大的时候，无所谓
- 当$d_k$很大（512）的时候，即**两个向量比较长的时候，点积的结果值就会比较大（或者比较小），当点积值比较大的时候，相对的差距就会变大，则最大的哪个值在softmax后就会更接近于1，剩下的值就会更加接近于0，这时候算梯度就会非常的小，跑不动**

> 因为softmax的最后结果是希望预测值，置信的地方尽量靠近1，不置信的地方尽量靠近0，这样就差不多收敛

$A'_{n×d_v}$的每一行就是所需要的输出

- 然后对结果的每一行做softmax，每一行之间是独立的

![Scaled Dot-Product Attention](Attention is all you need.assets/Scaled Dot-Product Attention.png)

#### Msasked

**问题**

- query向量和key向量长度都是相同的，且在时间上是可以对应起来的

- **对于$query_t$，应该只看到$key_1$到$key_{t-1}$部分的内容，而不应该看到$key_t$和之后的内容**

- 注意力机制在训练的时候每次都会看到所有的内容，从$key_1$到$key_n$

所以通过一个**带掩码的注意力机制**，来避免**训练的时候**解码器预测第t个时刻的输出的时候看到t时刻后面的内容

- 仍然会全部计算出k1到kn的内容
- 使用mask操作， 对于t时刻的$query_t$和$key_t$和之后的**计算值，换成非常大的负数**
  - 这样在进行softmax归一化时，对应的权重值就会变成0
- **从而保证训练和预测的行为是一致的**   

> 将$QK^T$的**上三角元素标为负无穷大**
>

#### Multi-Head Attention

与其做一个单个的attention

1. **不如将query向量和key-value向量对投影到一个低维的空间，然后做h次的attention**
   1. layer层就是投影到低维度
   2. **有h个头就是h次计算**
2. 然后将h次的attention的输出合并，再投影回高维的空间得到最终输出
   1. 即将h个头的输出合并
   2. concat层就是最后的投影

**目的**

- Dot-Product Attention没有什么可以学的参数，具体函数就是内积
- 有时候为了识别不一样的模式，希望有一些不一样的计算方法
- **所以先让query向量和key-value向量对投影到一个低维的空间，这个对应的线性变换的W矩阵是可以学的**
- **给了h次机会（h个头），希望能学到不一样的投影的方法，使得在投影进去的那个度量空间中能匹配不同的模式** 

![Multi-Head Attention](Attention is all you need.assets/Multi-Head Attention.png)

#### Self-Attention

**输入的query向量和key-value向量对其实是一个东西变换而来，所以叫自注意力机制**

> 对应架构图中input embedding进过position encoding后变成三个输入

### Encoder-Decoder传递Attention

连接的attention层不再是自注意力层

- **Encoder的输出作为key向量和value向量传入Decoder中**
- **Decoder的masked attention的输出作为query向量，然后和Encoder的输入key向量和value向量一起作为输入**

由于**attention层的输出结果是value向量的加权和，所以连接的attention层的输出就是Encoder传入的value向量的加权和**

- 权重就由Decoder的query向量和Encoder的key向量得出，即**输出取决于Decoder的query向量和Encoder的key向量的相似度**

> 根据Decoder的query的不同，会在当前Encoder输出的key中挑选相似的东西

![连接attention层](Attention is all you need.assets/连接attention层.png)

### Feed Forward NetWorks

**Position-wise Feed-Forward Networks**

- a fully connected feed-forward network
- 编码器和解码器的每一层都含有一个FNN

**Position-wise**

- attention层输出是一个序列，其中每一个向量就是一个position，将这个相同的MLP对每一个向量作用一次

$$
FFN = max(0,xW_1+b_1)W_2+b_2
$$

其中

- $xW_1+b_1$是一个线性层
- $max(0,xW_1+b_1)$是一个ReLU激活层
- $(max(0,xW_1+b_1))W_2+b_2$是另一个线性层

因为attention层的每一个输出都是一个512的向量

- 所以$x$就是一个512维的向量

- 然后$W_1$会将$x$投影成一个2048维度的向量
- 由于有残差连接，所以最后还需要有$W_2$去将2048维度的向量重新投影回512维度

**就是一个单隐藏层的MLP，中间的隐藏层将输入扩大4倍，最后的输出层再缩小回去**

因为在经过attention层处理后，每一个输出的向量已经抓取出序列中所想要的信息了

- **所以在经过MLP做投影的时候，因为每一个向量已经包含所需的信息，每个MLP只要在对每个单独的向量独立做运算就可以了**

> 做空间转换

**RNN和Transformer区别**

![RNN和Transformer区别](Attention is all you need.assets/RNN和Transformer区别.png)

### Embeddings and Softmax

输入是一个个token，需要将其映射成一个向量

- Embedding就是给任意一个token，**学习将一个长为d=512的向量来表示它**
- 编码器和解码器的输入都要经过一个Embedding，softmax前面也需要一个Embedding
  - 并且这三个Embedding的权重是一样的，方便训练
- 权重还会乘以一个$\sqrt{d_{model}} = \sqrt{512}$
  - 因为在学习Embedding的时候可能会将一个向量的L2L学成一个相对比较小的值，比如1，即不管维度多大，最后值都等于1
  - 维度一大，权重值就会变小，但是Embedding后还需要加上Positional Encoding的向量，所以将权重乘以$\sqrt{d_{model}}$后使得两者的维度差不多

### Positional Encoding

**attention是不会有时序信息的**

- 输出是value向量的加权和，权重是query向量和key向量的距离（相似度），**都与序列位置信息无关**
- 所以给定一个序列，将其中的向量位置打乱，得出的结果顺序会变，但是值是不会变的

> RNN本身就是一个时序的处理，将上一个时刻的输入作为下一个时刻的输出

因为输入序列无论怎么打乱顺序，输出值是一样的，所以选择**在输入值里面直接加入时序信息的值**

- attention的输入是在嵌入层表示的一个d=512长度的一个向量
- **所以现在依旧使用一个d=512长度的一个向量来表示一个数字，代表向量的位置**

> 因为计算机一个数字可以认为是一个**长度为32位的向量**来表示的

将**这两个向量相加就表示加入了时序数据的输入**

- 因为Positional Encoding的函数是一个cos和sin的函数，所以它的值是在-1和+1之间抖动的
- 所以将结果乘以一个$\sqrt{d_{model}}$，所以每个位置数字也差不多在-1和+1之间抖动

## 优点

- 计算复杂度
- 顺序的计算：越少越好，越少并行度越高
- 信息从一个数据点走到另一个数据点要走多远：越短越好

Self-Attention（restricted） 

- **query只和最近的r个key做计算**
- 距离比较远的两个点需要走几步才能过来

> n个维度为d的query
>
> - n：序列长度
> - d：向量长度

![比较](Attention is all you need.assets/比较.png)

没有太多东西可以调参

- $N$是多少层
- $d_{model}$是token表示成的向量的长度
- $h$是指有多少个头

传统seq2seq模型最大的问题在于

- **将Encoder端的所有信息压缩到一个固定长度的向量中**，并将其作为Decoder端首个隐藏状态的输入，来预测Decoder端第一个单词（token）的隐藏状态，**在输入序列比较长的时候，这样做会损失Encoder端的很多信息**
- 而且这样一股脑的把该固定向量送入Decoder端，**Decoder端不能够关注到其想要关注的信息**
- 并且**模型计算不可并行，会耗费大量时间**

Transformer

- **基于Encoder-Decoder架构**，抛弃了传统的RNN、CNN模型，**仅由Attention机制实现**，解决了输入输出的长期依赖问题
- **Encoder端是并行计算的，训练时间大大缩短，减少了计算资源的消耗**
- self-attention模块**让源序列和目标序列首先自关联起来**，这样源序列和目标序列**自身的embedding表示所蕴含的信息更加丰富**
- 后续的FFN层也增强了模型的表达能力
- Muti-Head Attention模块使得Encoder端拥有并行计算的能力，**Decoder端仍旧需要串行计算**

# Transformer计算细节

## Self-Attention

### 矩阵计算

$XX^T$

其中
$$
X = (x_1^t,x_2^t,x_3^t)\\
假设：x_1 = [1,2,1,2,1]^T,x_2 = [1,2,1,1,1]^T,x_3 = [2,1,1,2,1]^T\\
只计算x_1:x_1X^T = x_1(x_1^t,x_2^t,x_3^t)
\left[\begin{matrix}1&2&1&2&1\end{matrix}\right]
\left[\begin{matrix}1&1&2\\2&2&1\\1&1&1\\2&1&2\\1&1&1\end{matrix}\right]
=[11,9,10]
$$
矩阵计算可以看出行向量和列向量的**内积**

**内积的几何意义**：是一个向量在另一个向量上的投影

- **投影的值大，说明两个向量的相关度高，如果两个向量夹角是九十度，内积为0，即这两个向量线性无关，完全没有相关性**

> 所以[11,9,10]表示x1和(x1,x2,x3)的相关性，和自己的相关性一定是最大的

### Softmax计算

$Softmax(XX^T)$

> **hardmax**：求数组所有元素中值最大的元素，只选出其中一个最大的值，但是往往在实际中这种方式是不合情理的

Softmax的含义就在于不再唯一的确定某一个最大值，而是**为每个输出分类的结果都赋予一个概率值**，**表示属于每个类别的可能性**

**归一化**

- 将多分类的结果输出值转换为**范围在[0, 1]和为1的概率分布**

- **每一个向量和其他向量的结果进过归一化处理后得到的各个概率就是应该分配的注意力**

**指数函数**

- 指数函数曲线呈现递增趋势，斜率逐渐增大
- 所以**x的很小的变化，可以导致y的很大的变化**，所以指数函数能够将**输出的数值拉开距离**
- 在使用反向传播求解梯度更新参数的过程中，**指数函数在求导的时候比较方便**

**计算**

- 指数函数值可能会非常大
- 为了避免求$e^x$出现溢出的情况，一般**需要减去最大值**，即$e^{x-max}$

> softmax([11,9,10] = [0.67,0.09,0.24])

**损失函数**

- 使用Softmax函数作为输出节点的**激活函数**的时候，一般**使用交叉熵作为损失函数**
- 由于Softmax函数的数值计算过程中，很容易因为输出节点的输出值比较大而发生数值溢出的现象，在计算交叉熵的时候也可能会出现数值溢出的问题

> 为了数值计算的稳定性，TensorFlow提供了一个统一的接口，将Softmax与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常

### 加权求和

$Softmax(XX^T)X$

矩阵乘法归一化的结果再和矩阵相乘，此时的结果的每一个行向量都和X的维度dx相同

- 由计算方式可以看出，**每一个维度的数值（每一个行向量的分量）都是由三个向量在这一维度的数值加权求和得到**
- 权重就是softmax后得到的相关性
  - 即Query与Key作用得到attention的权值
- **这个新的行向量就是x1向量经过注意力机制加权求和之后得到的表示**，即权值作用在Value上得到attention值


$$
Softmax(XX^T)X\\
只计算x_1:Softmax(x_1X^T)X = 
\left[\begin{matrix}0.67&0.09&0.24\end{matrix}\right]
\left[\begin{matrix}1&2&1&2&1\\1&2&1&1&1\\1&1&1&2&1\end{matrix}\right]
=\left[\begin{matrix}1&1.76&1&1.91&1\end{matrix}\right]
$$

### QKV模型

需要**通过训练来找到一个加权**，因此最少需要给每个input定义一个tensor

- 但一个tensor，关系之间就是对偶的，只有一个tensor也无法存放计算的结果
- 所以至少给每个input定义2个tensor，Q和K
  - **Q是自己用的，用Q去找自己和别人的关系**
  - **K是给别人用的，用自己的K去处理别人的Q**
- 两个tensor也不够，找到了关系并没有使用，还要对input进行加权，和input直接加权比较生硬，所以再给每个input定义一个tensor，V
  - **V也是对input做变换得到的，所以等于对input又加了一层可以学习的参数，使得网络具有更强的学习能力**

> 通过query和key的相似性程度来确定value的权重分布的方法被称为scaled dot-product attention

Q，K，V矩阵本质上都是矩阵input X的**一个线性变换**
$$
Q = W^qX\\
K = W^kX\\
V = W^vX
$$
不直接使用矩阵X来进行计算而对它进行线性变换是为了提高模型的拟合能力，因为**W矩阵是可以通过训练优化的**

> Q，K，V思想最早来自于Memory Networks
>
> Memory Networks是一种思路：使用外部的一个memory来存储**长期记忆**信息，因为当时RNN系列的模型使用final state存储的信息，序列过长就会遗忘到早期信息

### 缩放因子

$\sqrt{d_k}$

- 假设Q，K中元素的均值为0，方差为1，则$A = Q^TK$的元素均值为0，方差为d
- 当d变得很大时，A的元素的方差也会很大，那么softmax(A)的分布就会趋于陡峭
  - 分布的方差大，分布集中在绝对值大的区域
- A中每个元素缩放后，方差又变为1，则softmax(A)的分布与d无关了

### 位置编码

**对self attention来说，它对每一个input vector都做attention，所以没有考虑到input sequence的顺序**，它缺少了原本的位置信息

- 没有说像RNN那样**后面的输入考虑了前面输入的信息**
- 也没有考虑**输入向量之间的距离远近**

对比来说，LSTM是对于文本顺序信息的解释是输出词向量的先后顺序，而self attention的计算对sequence的顺序这一部分则完全没有提及，**打乱sequence的顺序，得到的结果仍然是相同的**

> Positional Encoding是transformer的特有机制，弥补了Attention机制无法捕捉sequence中token位置信息的缺点
>

位置信息编码位于encoder和decoder的embedding之后，block之前

- 为每个位置的输入都设定一个独立的**位置向量**ei
- 再将位置向量加到输入向量上作为最终的输入

## Muti-Head attention

Muti-Head attention的每个头**关注序列的不同位置**，增强了Attention机制关注**序列内部向量之间**作用的表达能力
$$
Q_i = W_i^qX\\
K_i = W_i^kX\\
V_i = W_i^vX
$$
通过权重矩阵$W_i^{q,k,v}$将Q，K，V分割，因为$W_i^{q,k,v}$各不相同，所以计算的Qi，Ki，Vi也都不相同

- 即 **每个头关注的重点不一样**

Attention是将query和key映射到**同一高维空间中去计算相似度**，Multi-head Attention把query和key映射到**高维空间α的不同子空间**（α1，α2，α3...）取计算相似度，最后再将各个方面的信息综合起来

> 类比CNN中同时使用**多个卷积核**的作用，直观上讲，多头注意力**有助于网络捕捉到更丰富的特征/信息**

## Mask Muti-Head attention

mask表示掩蔽，它对某些值进行掩盖，使**其在训练时参数更新时不产生效果**

Transformer模型里面涉及两种mask，分别是padding mask和sequence mask

- padding mask在所有的scaled dot-product attention里面都用到
- sequence mask只有在**decoder**的self-attention里面用到

### padding mask

**padding mask实际上是一个张量**，每个值都是一个Boolean，值为false的地方就是要进行处理的地方

因为**每个批次输入序列长度是不一样的**也就是说，**要对输入序列进行对齐**

- 如果输入的序列太短，给较短的序列后面**填充0**
- 如果输入的序列太长，截取左边的内容，**把多余的直接舍弃**

因为这些填充的位置其实是没什么意义的，所以**attention机制不应该把注意力放在这些位置上**，所以需要进行一些处理

- 把这些位置的值加上一个非常大的负数（负无穷），这样经过softmax，这些位置的概率就会接近0

### sequence mask

**实际运算**

- 产生一个上三角矩阵，下三角的值全为0
- 把这个矩阵作用在每一个序列上，就可以达到目的

## Feed Forward NN

前馈神经网络/全连接神经网络

- FNN为Encoder引入**非线性变换**（ReLU激活函数），变化了attention的输出空间，增强了模型的拟合能力

## Residual Network

> 残差网络是深度学习中一个重要概念

**残差连接可以使得网络只关注到当前差异的部分**

- 在**神经网络可以收敛的前提下**，随着网络深度的增加，网络表现先是逐渐增加至饱和，然后迅速下降，这就是经常讨论的**网络退化**问题

- 在Transformer模型中，Encoder和Decoder各有6层，为了**使当模型中的层数较深时仍然能得到较好的训练效果，模型中引入了残差网络**

## Linear & Softmax

Decoder的输出会经过一个线性变换和softmax层

- Decoder最后的输出是一个实数向量，需要把浮点数对应成一个单词，这便是线性变换层的目的

**线性变换层是一个简单的全连接神经网络**

- 它可以**把Decoder产生的向量映射到一个比它大得多的、被称作对数几率（logits）的向量里**
- 假设模型从训练集中学习一万个不同的英语单词（即模型的输出词表），则对数几率向量为一万个单元格长度的向量
- 每个单元格对应某一个单词的分数，就相当于**做vocaburary size大小的分类**
- 接下来的Softmax层便会把那些分数变成概率，**概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出**

