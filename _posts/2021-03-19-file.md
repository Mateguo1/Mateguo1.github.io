---
layout: post
title: file
categories: file processing
description: Normal processing operations to files
keywords: file
---

# 文件处理常用操作：

本文整理了一些本人常用的文件的操作方法

## 1、CSV文件：

### （1）文本文件：

因为csv文件的本质就是使用了“ ，”作为分割符的文本文件，因此也可以是使用open方法进行读取等一些列操作，虽然这种方法比较简单，但是大多数操作需要自己手动写出来，不是很方便。

#### 读取：

```python
with open("file_name.csv","r") as f:
    data = f.readlines()
```

#### 写入：

```python
with open("file_name.csv","w") as f:
    for i in data:
        f.write(i+"\n")#添加了换行符
#write是写入字符串，返回字符串长度
#另一种方法f.writerlines()，写入字符串列表，但同样换行也需要加入换行符
```

### （2）csv标准库：

使用csv标准库进行操作

#### 读取：

```python
import csv
with open("file_name.csv","r") as f:
    data = csv.reader(f)
```

#### 写入：

```python
# 通常
import csv
with open("file_name.csv","w") as f:
    writer = csv.writer(f)
    row_names = ["表头","表头","表头"]
    writer.writerow(row_names)#列名
    for i in data:
        writer.writerow(i)#不 需要加换行符
    # 上面两行，可替换成
    # writer.writerows(data)
```

```python
# 结构化数据（字典），爬虫
import csv
with open("file_name.csv","w") as f:
    writer = csv.writer(f)
    row_names = ["表头1","表头2","表头2"]
    writer = csv.DicWriter(f,fieldnames=row_names)
    writer.writeheader()
    writer.witerow(["表头1":"0000","表头2":"0000","表头2":"0000"])
```

### （3）pandas：

很推荐，方便，这个具体参考pandas的介绍吧

```python
import pandas as pd

csvframe = pd.read_csv("file_name.csv")
# 可替换为，注意分隔符
# csvframe = pd.read_table("file_name.csv",",")
```

## 2、文件复制：

直接看代码

```python
from shutil import copyfile
copyfile("source_file","target_file")
```

