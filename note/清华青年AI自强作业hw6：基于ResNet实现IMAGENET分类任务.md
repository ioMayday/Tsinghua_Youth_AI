@[toc](清华青年AI自强作业hw6：基于ResNet实现IMAGENET分类任务)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c3c2e4b7dae44bef909a63d75eeb227f.jpeg)

> 一起学AI系列博客：[目录索引](https://blog.csdn.net/qq_17256689/article/details/130910780)

## 简述

---

hw6作业为基于ResNet模型，并利用VGG标准模块和GoogleNet中的inception模块对IMAGENET数据集进行20类分类。模型输入图像尺寸为`299*299`，输出为softmax后的20分类。

观察参考代码发现需要使用IMAGENET处理好后的数据ILSVRC2012_20_tfrecord，由于缺乏实验数据，本次作业不进行实战，只对TF1.x版本的参考代码进行思路梳理学习。

- 相应实现源码见代码仓：[https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework](https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework)
- 相关keras使用指导：[https://keras.io/zh/getting-started/sequential-model-guide/](https://keras.io/zh/getting-started/sequential-model-guide/)

## 作业实现

---

- project文件架构
    - test：测试一张图是否正确
        - read_data(获取数据及数据增强等预处理)
        - EX6_NET（核心模型定义）
            - utils(参数设置模块)
    - finetune：预训练模型调优
        - read_data
        - EX6_NET
    - 说明
        - 仅分20类
        - 训练集样本：26000
        - 验证集样本：1000

从代码中，可以判断模型应该是基于ResNet，其中`EX6_NET.py`是核心网络搭建文件。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1c6f5aa0e01b4e16ba1d5f3dbe91ee32.png)

> ResNet网络模型结构


代码中模型结构为：

- 299 x 299 x 3, 卷积层，3x3
- 149 x 149 x 32
- 147 x 147 x 32
- 147 x 147 x 64 # 加最大池化3x3，有分支，resnet直连
- 73 x 73 x 160 # 加最大池化3x3，有分支，resnet直连
- 71 x 71 x 192 # 加最大池化3x3，有分支，resnet直连
- 35 x 35 x 384 # 两个inception模块
- 35 x 35 x 384 # Reduction block卷积加下采样
- 17 x 17 x 384 # 普通卷积3x3
- 8 x 8 x 300   # 普通卷积3x3

该模型中，用到VGG里的标准模块+堆叠（3x3、1x1、1xn、nx1），GoogleNet里的inception，ResNet中的（Branch_0/Branch_1部分）分支直连特性。



![在这里插入图片描述](https://img-blog.csdnimg.cn/45dd8627e0f94cdd853e7d0527851563.png)

> GoogleNet中Inception模块


![这里插入图片描述](https://img-blog.csdnimg.cn/9a2189de20ef489b84bcc17067294d87.png)

> VGG16中标准卷积模块堆叠加深

![在这里插入图片描述](https://img-blog.csdnimg.cn/6a51371d7b2e4999b672e862460df459.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0e7e71608bbf4b9e9ec2f51db0b100cc.png)

> ResNet短接直连特性



## 相关链接																								

---

> 1. 文科生都能零基础学AI？清华这门免费课程让我信了，[link](https://blog.csdn.net/qq_17256689/article/details/123290351)
> 2. 清华青年AI自强作业hw2：线性回归预测，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 3. 清华青年AI自强作业hw3_1：用线性回归模型拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280024)
> 4. 清华青年AI自强作业hw3_2：前向传播和反向传播实战，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 5. 清华青年AI自强作业hw3_3：用NN网络拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280109)
> 6. 清华青年AI自强作业hw4：基于DNN实现狗狗二分类与梯度消失实验，[link](https://blog.csdn.net/qq_17256689/article/details/131424142)
> 7. 清华青年AI自强作业hw5：基于CNN实现CIFAR10分类任务，[link]( https://blog.csdn.net/qq_17256689/article/details/131501974)
