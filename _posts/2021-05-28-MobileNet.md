---
layout: post
title: MobileNet
categories: CNN
description: MobileNet
keywords: CV, CNN
---

# MobileNet V1：

​	MobileNet是2017年由Google团队提出的一种专注于嵌入式、移动端设备的轻量级卷积神经网络。相比传统卷积神经网络，在牺牲一小部分精度的前提下，极大程度地减少模型运算的参数量，其中的两个亮点分别是使用了深度可分卷积和增加了超参数α、β。

![image-20210528191936856](/assets/img/image-20210528191936856.png)

​        下图为MobileNet V1的网络结构：

![image-20210528214705588](/assets/img/image-20210528214705588.png)

## Depthwise Separable Convolution：

​	MobileNet V1通过使用深度可分卷积，有效的减少了模型运算过程参数数目，下图为常规卷积和DW卷积之间的参数运算量对比，假设二者输入均为3通道的RGB图像，输出均为四个feature map：

![image-20210525085320164](/assets/img/常规卷积.png)

计算结果为：3×3×3×4=108

​	DW卷积分为两步进行，首先是下图所示的Depthwise convolution：

![image-20210525085320164](/assets/img/Depthwise.png)

​	然后是下图所示的Pointwise convolution：

![image-20210525085320164](/assets/img/pointwise.png)

计算结果为：3×3×3=27 + 1×1×3×4=12 = 39 << 108;

## α、β：

​	我们首先来把整个卷积计算的公式写出来，这里为了显示方便，我将对应变量都标注到了常规卷积的图片上

​       ![image-20210525085320164](/assets/img/解释.png)

​	下面显示的是DW卷积的参数计算公式：

![image-20210528215600759](/assets/img/image-20210528215600759.png)

​	α指的是**宽度乘子（ Width Multiplier ）**，主要是为了更小、计算代价更少的模型，其取值范围为 (0,1]，将输入通道尺寸由M变为αM，输出通道尺寸由N变为αN。

![image-20210528205109391](/assets/img/image-20210528205109391.png)

​	在添加了α之后，DW卷积的参数计算公式如下：

![image-20210528215625240](/assets/img/image-20210528215625240.png)

​	β指的是**简式显示（ Reduced Representation ）**，，其取值范围为 (0,1]，通过改变输入尺寸，来减少神经网络的计算代价。

![image-20210528205128060](/assets/img/image-20210528205128060.png)

在添加了β之后，DW卷积的参数计算公式如下：

![image-20210528215635800](/assets/img/image-20210528215635800.png)

# MobileNet V2：

​	MobileNet V2在2018年由Google团队提出，相比于MobileNet V1，主要是提出了有线性瓶颈的逆残差结构，其整体网络结构如下所示：

![image-20210525085320164](/assets/img/MobileNet.png)

## Inverted Residuals with Linear Bottlenecks：

![image-20210528211607757](/assets/img/image-20210528211607757.png)

上图所示为残差模块和逆残差模块之间的对比，二者每一层的对比如下：

| Residual block | Inverted residual block |
| :------------: | :---------------------: |
|  1×1卷积降维   |       1×1卷积升维       |
|    3×3卷积     |         3×3卷积         |
|  1×1卷积升维   |       1×1卷积降维       |

还引入了ReLU6，其公式为：***ReLU(x) = min(max(0,x),6)*** 。

其与ReLU之间的对比，直观展示如下：

![img](/assets/img/wps2.jpg)

此外，有线性瓶颈的逆残差结构的最后一层的激活函数替换为了线性激活函数。

# MobileNet V3：

​	MobileNet V3在2019年由Google团队提出，它保留了前两代的深度可分卷积、超参数α和β以及有线性瓶颈的逆残差结构，加入了基于压缩-激励结构（squeeze - excitation structure)的轻量级注意力机制更新了Block，并对较为耗时的Last Stage重新进行了设计，网络结构如下图所示：

![image-20210528214326323](/assets/img/image-20210528214326323.png)

## Bneck:

更改后的block结构如下图所示：

![image-20210528212610295](/assets/img/image-20210528212610295.png)

注意力机制借鉴了我们通过眼睛在观察事物的时候，往往会聚焦在某些物体上，从而过滤掉一些无用的信息，因此注意力机制的作用就是通过调整每个通道的权重不同，从而实现对输入数据的每个部位的关注度不同，从而提高模型对关键特征的识别能力，其应用位置如下图所示：

![image-20210528212657728](/assets/img/image-20210528212657728.png)

该结构的简化举例如下图所示：

![image-20210525085320164](/assets/img/SE.png)

其中用到的新的激活函数hard-sigmoid，公式如下：

![image-20210529074923076](/assets/img/image-20210529074923076.png)

## Last Stage:

简化前后对比，清楚看到其中减少了层结构，最终的效果能够减少模型运行的时间为7ms大约占据一次运行时间的11%：

![image-20210528213446844](/assets/img/image-20210528213446844.png)

其中用到的新的激活函数hard-swish，公式如下：

![image-20210528222414112](/assets/img/image-20210528222414112.png)

