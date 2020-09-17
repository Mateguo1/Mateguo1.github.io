---
layout: post
title: Detailed explanation of Batch Normalization
categories: Computer Version
description: Batch Normalization
keywords: Batch Normalization
---

# Batch Normalization:

Batch Normalization was proposed by Google team in Paper "Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift" in 2015. The method could accelerate the convergence of the network and improve the accuracy. Although there are many related articles on the Internet, they rarely focus on how BN really works.

In this article, I will show you the following points:
1) principle of BN
2) why use BN
3) point for attention in using BN

## 1.  Principle of BN:

In the process of image Preprocessing, we usually standardize the image, which can accelerate the convergence of the network. As shown in the figure below, for conv_1, the input is a feature map satisfying a certain distribution. However for conv_2, the input feature map may not satisfy a certain distribution law. The purpose of BN is to make the feature map satisfies the distribution law of mean value 0 and variance 1.

![BN1](/assets/img/BN1.png)

Next, let's take a look at the statement in the original paper, that is in blue box below.

![image-20200917113852396](/assets/img/image-20200917113852396.png)

 For example, we can assume that the input ***x*** is a RGB three channer color image, ***d*** means the input image's channel, that is , d = 3. And now let's take a look at the formula given in the article.

![BN2](/assets/img/BN2.png)

首先是对一个Batch size里面的所有样本进行求均值、方差，然后是对于每一个样本先减去均值，然后再除以标准差，注意这里的$\epsilon$一般就是设置一个非常小的值，这是为了防止方差为零的情况出现。然后最后再通过$ \gamma$对方差进行调整，通过$\beta$对均值也就是中心点进行调整。这里的$ \gamma$和$\beta$的初值分别是1和0，也就是上面所说的feature map所要满足的规律，当然这里的$ \gamma$和$\beta$是要通过反向传播来进行调整的，这是因为上述规律的效果可能并不是最好的。

接下来我借用一位大佬的博客内容，举例解释下如何计算方差$\sigma^2$和均值$\mu$：

![BN2](/assets/img/BN2.png)