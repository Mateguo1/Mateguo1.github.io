---
layout: post
title: Transformer
description: deep-learning
keywords: deep-learning
---

# Transformer:

[TOC]

今天新开一个坑，主要是为了ViT（虽然之前好多坑都没填完，哈哈哈哈），首先《Attention Is All You Need》论文链接：https://arxiv.org/pdf/1706.03762.pdf。

接下来，先整理下李宏毅老师的ML课上讲的transformer，课件链接：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/seq2seq_v9.pdf，然后这部分主要是针对我听完课程之后的一些个人的思路总结，顺序上可能和课件不太一样，因此还是很建议大家直接去看李老师的ML课程视频的，讲的可太好了！！！（ps：由于是学习总结，所以会借用大量李老师课件里的图），主要还是理论部分，代码的坑等我过几天再填吧。

按惯例，直接来吧，先上网络结构，下图是原论文中给出的结构图片，其中可以主要分为encoder和decoder（这俩×N），然后还有一个Embedding和Positional Encoding，下面就简单的梳理一下Transformer的整体思路和相关内容。

![Transformer_0](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/Transformer_0.jpg)

## 1. Transformer：

### 1.1 Embedding：

这个其实就是将每一个词变成了词向量

### 1.2 positional encoding：

由于transformer是为了处理机器翻译的，这里面是包含时序关系的，而self-attention对于时序关系是考虑不到的，而对比于RNN的那种在模型内部可以考虑到前面的部分特征而言，transformer使用了positional encoding在输入的时候就对输入的向量集合加入了时序关系。

### 1.3 encoder：

![image-20211130215448550](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130215448550.png)

首先来看encoder结构，很明显可以看到encoder前面是有一个N×（论文里面是给出了N=6），而这N个encoder的网络结构是一样的，但是其中的参数是不一样的，通过下图可以看到一个block的结构可以分解为右半部分那样。

![image-20211130221911379](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130221911379.png)

首先Add&Norm，这两步操作可以通过下图来展示，分别为residual和layer normalization，其中残差结构在ResNet中提到过，这里就不再赘述了。

![image-20211130222242028](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130222242028.png)

### 1.3.1 layer norm：

对比batch norm（每一个特征，减掉均值，除以方差，均值0，方差1），layer norm（每一个样本），二维是转置batch norm转置即可，而三维的话，切出来的是两个平面，一个原因：时序数据的样本长度变化，变化大，均值方差变化大，做预测全局均值方差，遇到一个特殊的样本就可能不那么好用，而layer是在样本里面算的，就没有这样的问题了。之后有的论文对layer norm的梯度方面进行了分析，这里就不再深入讲解了（留个坑）。

### 1.3.2 attention：

其中涉及到了self-attention的部分，下图是原论文中给出的新的注意力方法Scaled Dot-Product Attention，下面我先整理一下李宏毅老师的课里讲的、详细的self-attention（ppt：https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf），所以会再多借用下李老师的图，hhh，但是仅限于整理，具体内容还是建议大家去看视频学习的。

![](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211201114128528.png)

#### 1.3.2.1 self-attention：

![image-20211130215940213](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130215940213.png)

首先解释下上图中的变量表示的意义，输入 $ A=[a_1, ....., a_n] $ ，每一个向量对应有一个$q^i, k^i, v^i $，其中q是query，k是key，v是value，是分别根据$ W^q\cdot{a^i}, W^k\cdot{a^i}, W^v\cdot{a^i} $得到的，而这$ W^q, W^k, W^v$就是我们要学习的参数，然后$\alpha'_{i,j}=q^i\cdot{k^j}$，主要是用来表示$a^i$和$a^j$的相关程度（内积值越大，余弦值越大，相似度越高），最后$b^i=\sum_{i} \alpha'_{i,j} \cdot v^i$，这里提示个小点，i是作为q，而j是作为k。

#### 1.3.2.2 Scaled Dot-Product Attention：

![image-20211201114859810](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211201114859810.png)

![image-20211201115037227](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211201115037227.png)

结合原论文中给出的上图结构和公式，来分析下这个attention，其实相比上面的1.3.2.1中attention多除了一个$\sqrt{d_k}$，这个d就是dimension of k，而其实最后q、k、v最终的维度应该是相同的，然后文章中提到了两种常用的注意力函数分别是additive attention和dot-product（这个就是1.3.2.1中的），当然了三者维度不同的话，也可以用additive attention，这里就不再详细展开了（再挖个坑）。

#### 1.3.2.3 Multi-head Self-attention：

![image-20211201115801372](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211201115801372.png)

至于transformer中的Mutil-head Self-attention，也很简单，上图是原论文中的结构，这里说明下用它的一个原因，因为Scaled Dot self-attention说白了，没有可以学习的参数，只是通过点积来计算，因此为了更好的对应不同问题的特征进行分析，就用了不同的线性映射，而每一个线性映射都可以并行运算，原文中是用了8 heads。

![image-20211130221216612](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130221216612.png)

上图是2 heads的例子，其实就是将$q^i, k^i, v^i $以及 $b^i$ 变成了$q^{i,k}, k^{i,k}, v^{i,k}, b^{i,k}$。

#### 1.3.4 position-wise Feed-Forward：

这个其实就是通过一个1×1的卷积核来升维，再一个1×1的卷积核来降维，中间再加一个ReLU，公式如下图所示。

![image-20211201121207977](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211201121207977.png)

### 1.4 decoder：

自回归，输入是上一层的输出

![Transformer_1](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/Transformer_1.jpg)

其实从上图中，可以看出decoder的结构貌似和encoder有一部分很像，只是下面多了框框中的部分，Masked Multi-Head Attention + Add&Norm。

### 1.4.1 Masked Multi-Head Attention：

首先来看，其中的Masked Multi-Head Attention，对比下图attention和下下图所示的Masked attention其实很简单，因为当前的输入是有序列的$a^i$，而原始self-attention的操作产生$b^i$时，会考虑所有的$a^i$，Masked attention是不再考虑包括$a^{i+1}$在内的右边的输入。

![image-20211130224248288](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130224248288.png)

![image-20211130223714932](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/image-20211130223714932.png)

### 1.4.2 ：

![Transformer_2](https://raw.githubusercontent.com/Mateguo1/Pictures/master/img/Transformer_2.jpg)

来看下上图中的很模糊的框框里面输入来源的对比吧，可以发现在decoder和encoder很相似那部分的输入实际来源是两个的，它的左边输入的key和value是来自于encoder输出，而右边输入的query是来自于decoder第一部分的输出，所以对比另外两个模块的self-attention的q、k、v都是来源于自己的输入而言，这个或许可以不叫成self-attention了吧。
