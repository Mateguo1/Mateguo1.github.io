---
layout: post
title: Image_Data_Amplification
categories: Image
description: Methods to amplify the Image Data
keywords: 扩增,图像
---

# 图像数据集扩增：

对于本次的比赛复赛的数据集不想过多评价了，想了想还是用初赛数据集进行扩充吧。因此就有了本篇关于图像数据集扩增入门级记录。

## 1、添加噪声处理

推荐使用skimage进行处理

```python
from skimage import io,util
#输入、输出路径
path_origin = ""
path_out = ""
def add_noise(filename):
    img = io.imread(path_origin+filename)
    '''
    参数较多下面就列出三个常用的
    img: 图像
    mode:‘gaussian’,‘localvar’,‘poisson’,‘salt’,‘pepper’,‘s&p’,‘speckle’
    seed: 整数
    '''
    img_out = util.random_noise(img,mode="",seed=1)
    io.imsave(path_noise+filename,img_out)
    
pics = os.listdir(path_origin)
for pic in pics:
    noise(pic)
print("done!")

#下面是skimage的显示图片
io.imshow(img)
plt.show()
```

## 2、直方图均衡化

