---
20200917202952layout: post
title: Detailed explanation of Batch Normalization
categories: Computer Version
description: Batch Normalization
keywords: Batch Normalization
---

# Batch Normalization:

Batch Normalization was proposed by Google team in Paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" in 2015. The method could accelerate the convergence of the network and improve the accuracy. Although there are many related articles on the Internet, they rarely focus on how BN really works.

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

First, caculate the mean value and variance of all samples in a batch size, and then subtract the mean value for each sample, and then divide it by the standard deviation. Note that the ![20200917203605](/assets/img/20200917203605.png)here is usually set to a vert small value, which is to prevent the occurrence of zero variance. Finally, adjust the square difference through ![20200917202838](/assets/img/20200917202838.png) , and adjust the mean value, that is, the center point, through![20200917202952](/assets/img/20200917202952.png) .The initail values of ![20200917202838](/assets/img/20200917202838.png) and![20200917202952](/assets/img/20200917202952.png) here are 1 and 0 respectively, which is the rule that feature map mentioned above must satisfy. Of course the ![20200917202838](/assets/img/20200917202838.png) and![20200917202952](/assets/img/20200917202952.png) need to be adjusted through back propagation, because the effect of the above law might not be the best one.

Next, I'll use a blog from a big shot to explain how to caculate the variance ![1600345427335](/assets/img/1600345427335.png)and the mean ![1600345478532](/assets/img/1600345478532.png) ：

![BN3](/assets/img/BN3.png)

The above figure shows us the calculation process of batch normalization with batch size 2, that is, two pictures. Suppose that feature1 and feature2 are the feature maps obtained from image1 and image2 after a series of convolution or pooling, the channel of feature is 2, so x1 and x2  is the data of channel in those batches.At last, you'll end up with two vectors, ![20200917202838](/assets/img/20200917202838.png) and ![20200917202952](/assets/img/20200917202952.png).

## 2. why use BN:

With the developement of Deep Learning, Dropout has been gradually replaced by BN in modern convolution architecture.I think there are three reasons for this as below:

1)BN also has the same regularization effect as dropout;

2)The regularization effect of dropout on convolution is limited. Comparesd with the fully connected layer, the training parameters of convolution layer are less, and the activation function can also complete the spatial transformation of features, so the regularization effect is not obvious in the convolution layer;

3)The full connection layer where dropout can play an important role is gradually replaced by global average pooling, which can not only reduce the model size, and also improve the performance of the model.

## 3. point for attention in using BN

1)The larger the batch size is, the better the performace of BN will be. A larger batch size means that![1600345427335](/assets/img/1600345427335.png)and ![1600345478532](/assets/img/1600345478532.png) will be closer to the mean and variance of the whole training set.

2)It's suggested that the BN layer be placed between the convolution layer and the active layer, and the bias needn't be used in the convolution layer, because it is useless. Refer to the following figure for reasoning, even if bias is used, the Result is same: ![1600345509035](/assets/img/1600345509035.png)

![BN4](/assets/img/BN4.png)