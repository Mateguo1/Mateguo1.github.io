---
layout: post
title: Inception
categories: DL
description: Inception
keywords: NN, Inception
---

# Inception:

Here is the hyperlink of the <a url="https://arxiv.org/pdf/1409.4842.pdf">GoogLeNet</a>.

Improved ability of object classification and detection: more powerful hardware, larger datasets and bigger models, new ideas, algorithms and improved network architectures.

They achieved a new network architecture which have increased depth and are able to make full use of the limited computional resources.

Deeper: introducing Inception module (increased network width) and increased network depth.
Drawbacks: 1) a larger number of parameters which makes the network more prone to overfitting; 2) increased use of computational resources.

Purpose of $1\times1$ convolution: as a dimension reduction method to remove computational  bottlenecks, so increasing the number of network layer from depth and width is feasible.
Solutions: introducing sparsity and replace the fully connected layers by sparse ones.

Here is a brief description of RCNN: it uses the low-level clues such as color and texture to locate the object and uses CNN to classify those loactions. 

The main idea of Inception Module is to find the optimal local construction and to repeat it spatially.
Some strategies they adopted: 1) increasing the ratio of $3\times3$ and $5\times5$ convolutions when moving to higher layers; 2) use  $1\times1$ convolutions and ReLU to reduce the dimensionality before $3\times3$ and $5\times5$.

Different image patch sampling methods: ??? what this means?

Other layers they used: 
The average pooling layer us before classifier improve the accuracy.
The Linear layer enables the network easily adapted to other label sets.
The dropout is still essential





