---
layout: post
title: Transformer
description: deep-learning
keywords: deep-learning
---

# Vision Transformer:

[TOC]

哈哈哈哈，接着来挖个ViT的新坑吧，首先上论文链接：https://arxiv.org/pdf/2010.11929.pdf，这篇论文里有很多的对比消融实验部分，我就不详细说了，主要还是说说模型的思路（我自己理解的）。

整体结构，首先来看原论文中给出的下图中的结构，原论文中说他们主要仍然采用transformer的结构，所以不难理解，整个结构主要分为三块Embedding、Transformer Encoder、MLP Head，下面就按这个顺序。

![image-20211203101814028](https://i.loli.net/2021/12/03/4C7sxgyfKlRSBP6.png)





## 1. Embedding：

transformer encoder 的输入要求为token，也就是一个[num_token,token_dim]的矩阵，对于图像而言[H, W, C]的矩阵而言，ViT将图像（224×224×3）分成了很多的patch（16×16×3），所以就可以算的最后num _patch就等于14×14=196，在经过一个linear projection，将patch变为token（1，16×16×3=768），然后tokens现在是（196，768），然后使用了一个专门用于分类的的cls，tokens现在就被concat成了（197，768），而我们知道图像之间是有顺序关系的，所以还需要加入一个position embedding（197，768），注意这里position和tokens之间是相加操作，所以最终得到的tokens‘ 大小仍然是（197，768），并且对于transformer的position而言ViT的是可训练的参数。

![image-20211203103822943](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20211203103822943.png)

这里提一下，代码中切割patch可以用conv16×16 Stride=16的卷积操作来实现，而在position embedding后还需要经过一步dropout(emd_dropout)才进入了encoder block里面。

## 2. Transformer Encoder：

![image-20211203112220105](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20211203112220105.png)

Encoder这里就和Transformer的encoder差不多了，MLP那里也很简单，就是Linear->GELU->Linear->Dropout，其中第一个Linear会把输入节点翻成4倍，按照上面大小的就是变成（196，3072），第二个Linear再把它缩小变回（196，3072）。

### 2.1 GELU：

全称：Gaussian Error Linerar Units，这是相关知识的整理，也不太明白到底原理是啥，这里先附上一个链接https://baijiahao.baidu.com/s?id=1653421414340022957&wfr=spider&for=pc，留坑。

函数形式$ GELU(x)=0.5x(1+tanh(\sqrt{2/\pi(x+0.044715x^3)})) $

## 3. MLP Head

MLP head之前和transformer encoder之后，是有一个layer norm的，这个没啥可讲的，就是Linear层。

最后来一个表简单的理解一下整个网络结构。

![image-20211203114949262](https://i.loli.net/2021/12/03/LgVdsojBCMHi7J3.png)
