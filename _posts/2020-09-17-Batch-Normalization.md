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

 For example, we can assume that the input ***x*** is a RGB three channer color image, 
***d*** means the input image's channel, that is , d = 3. And now let's take a look at the formula given in the article.

