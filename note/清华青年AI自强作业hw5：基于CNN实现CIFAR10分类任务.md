@[toc](清华青年AI自强作业hw5：基于CNN实现CIFAR10分类任务)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c3c2e4b7dae44bef909a63d75eeb227f.jpeg)

> 一起学AI系列博客：[目录索引](https://blog.csdn.net/qq_17256689/article/details/130910780)

   


## 简述

---

hw5作业为利用深度卷积神经网络实现CIFAR_10数据集十分类问题，帮助理解CNN的前向传播结构。

CIFAR-10是一个常用的彩色图片数据集，它有10个类别: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'。

其中，训练集样本5W张图，测试集样本1W张图。

![在这里插入图片描述](https://img-blog.csdnimg.cn/bce20b8dea4c42df8e59b752199cdf7e.png)

> 官网介绍：The CIFAR-10 dataset，[link](http://www.cs.toronto.edu/~kriz/cifar.html)

- 相应实现源码见代码仓：[https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework](https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework)
- 相关keras使用指导：[https://keras.io/zh/getting-started/sequential-model-guide/](https://keras.io/zh/getting-started/sequential-model-guide/)

## 作业实现

---

首先，根据课程作业上搭建的模型网络，得到总体算法流程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dcb52e29922e4e6bbd5b75a74c0ec3d9.png)

> 简单的CNN网络模型及具体参数

**算法流程**

- 输入图像
- 进行数据增强
- 两层卷积网络
    - 卷积计算
    - 加入偏置
    - 激活函数
    - 池化计算
- 将结果输出展开成1维向量
- 通过两层FC网络
- 通过softmax得到10分类结果

注意，卷积+池化+偏置+relu，这里我们整体算1层卷积网络。

**基本超参设置**

- batch-size: 256
- epoch：100
- lr:0.1
- learning_decay:0.001
- loss：交叉熵
- 优化方法：adam
- 模型层数：4
- batchnorm: yes
- 激活函数：relu
- 末端输出：softmax

**实现步骤**

- 本地调试
    - 先本地CPU跑通demo
- 服务器正式训练
    - 在服务器跑GPU
    - 先调小的epoch、batchsize、样本输入，保证gpu环境通的
    - 再调正式的参数进行训练



## 遇到的问题

---

**数据集100次epoch后loss就无法下降，预测集精度仅55%**

- 老师结果：70.18%
- 学生结果：87.35%

**现象分析**

- 是否训练次数不够高，欠拟合
    - epoch调大后，依然无济于事
- 查看样本大小，原始的：5w训练，最新的：10w训练，是否训练集问题？
    - 确认不是，最新的数据集本质也是5w张图片，只是数据增强后，增加了5w张图
- keras编译项loss还是二分类loss，应该改为categorical_crossentropy。
    - 确认有影响，导致模型loss无法正常收敛
- 是否学习率过大导致loss无法进一步收敛？
    - 确认有影响，学习率衰减设置过大0.8，导致lr很快为0，改为0.001，再后期可以精细化调节loss

**结论**

- 编译项loss类型选择错误
- 学习率设置过大

在分析过程中，同期实现了基于keras官网介绍的[模型训练demo](https://keras.io/zh/examples/cifar10_cnn_tfaugment2d/)，改进后测试精度达到：76%。主要做了以下修改：

- 网络卷积层数和feature个数不同
- 去除图片增强模块
- 基于colab环境运行，适配其keras版本

最终，基于课程上的网络模型训练，经过100个epoch得到的结果是：

- 训练集：99.97%
- 测试集：68%

说明有些过拟合，后期还需要增大正则化、减小epoch提前终止训练等方式来进行优化。

## 相关链接

---

> 1. 文科生都能零基础学AI？清华这门免费课程让我信了，[link](https://blog.csdn.net/qq_17256689/article/details/123290351)
> 2. 清华青年AI自强作业hw2：线性回归预测，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 3. 清华青年AI自强作业hw3_1：用线性回归模型拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280024)
> 4. 清华青年AI自强作业hw3_2：前向传播和反向传播实战，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 5. 清华青年AI自强作业hw3_3：用NN网络拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280109)
> 5. 清华青年AI自强作业hw4：基于DNN实现狗狗二分类与梯度消失实验，[link](https://blog.csdn.net/qq_17256689/article/details/131424142)
