---
layout: post
title: Image Augmentation
categories: Image
keywords: method
---

# Image Augmentation：

AlexNet中使用了图像增广（Image Augmentation）来获得更好的训练结果，该方法是通过使训练图像进行一系列的随机变化后，生成相似而不同的训练数据，从而增大训练集。此外，通过随即改变训练样本，还可以提高模型泛化能力。

下面就总结Pytorch中几个常用的方法：

```python
import torch
from torch import nn
import torchvision
from PIL import Image
# 读取一张图片
image = Image.open("file_name.jpg")
toTensor = torchvision.transforms.ToTensor()
image = toTensor(image)
# GPU加载图像
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img = image.to(device)
```

## 1. 图像翻转：

### 1.1 上下翻转：

```python
# 被翻转的概率，默认为50%
flip_aug = torchvision.transforms.RandomHorizontalFlip(p=0.8)
flip_aug(img)
```

### 1.2 左右翻转：

```python
# 被翻转的概率，默认为50%
flip_aug = torchvision.transforms.RandomVerticalFlip(p=0.8)
flip_aug(img)
```

------

## 2. 图像裁剪：

```python
# size：裁剪后的高宽被缩放到固定像素值
# scale：裁剪一个区域，面积为原始[0.1,1]倍
# ratio：上述区域的宽高比为[0.5,2]之间
shape_aug = torchvision.transforms.RandomResizedCrop(
    		(200, 200), scale=(0.1, 1), ratio=(0.5, 2))
shape_aug(img)
```

------

## 3. 图像颜色：

### 3.1 亮度：

```python
# 亮度：[1-0.5,1+0.5]
# 对比度：[1-0.5,1+0.5]
# 饱和度：[1-0.5,1+0.5]
# 色调：[-0.5,0.5]
color_aug = torchvision.transforms.ColorJitter(
    		brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
color_aug(img)
```

### 3.2 对比度：

```python
color_aug = torchvision.transforms.ColorJitter(
    		brightness=0.5, contrast=0, saturation=0, hue=0)
color_aug(img)
```

### 3.3 饱和度：

```python
color_aug = torchvision.transforms.ColorJitter(
    		brightness=0.5, contrast=0, saturation=0, hue=0)
color_aug(img)
```

### 3.4 色调：

```python
color_aug = torchvision.transforms.ColorJitter(
    		brightness=0.5, contrast=0, saturation=0, hue=0)
color_aug(img)
```

------

## 4. 混合使用：

```python
augs = torchvision.transforms.Compose([
    flip_aug, color_aug, shape_aug])
augs(img)
```

------

## 5. 训练模型时使用：

```python
#  这里以cifar10作为例子
import torch
from torch import nn
import torchvision

all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=4)
    return dataloader

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

train_iter = load_cifar10(True, train_augs, batch_size)
test_iter = load_cifar10(False, test_augs, batch_size)
```
