@[toc](清华青年AI自强作业hw3_3：用NN网络拟合MNIST手写数字分类)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c3c2e4b7dae44bef909a63d75eeb227f.jpeg)

> 一起学AI系列博客：[目录索引](https://blog.csdn.net/qq_17256689/article/details/130910780)

hw3_3：用NN网络拟合MNIST手写数字分类

- 体会神经网络设计和TF框架编程

- 对比hw3_1两者的模型、效果差异

## 实现过程

----

有了上篇博客：[清华青年AI自强作业hw3_1](https://blog.csdn.net/qq_17256689/article/details/131280024)的铺垫，本次任务只需在其基础上进行模型修改即可。

### 具体思路

1. 分类模式修改
2. 由二分类改为多分类模型（十分类）
3. 网络模型修改

### 多分类网络模型

1. 核心思路
    1. 用1个多分类模型处理0-9的分类
    2. 将网络叠加几层，加深加大
2. MNIST数据预处理(load)
3. 模型训练过程(train)
    1. 模型搭建
        1. 模型为深度神经网络
        2. 选取相应激活函数
4. 模型训练
    1. 前向推理与模型参数更新
    2. 按规律预测测试集的精度
5. 模型推理过程(predict)
    1. 在测试集上验证预测精度
6. 关键点
    1. 涉及到多分类数据标签的one-hot编码理解和手动处理

**神经网络模型说明**

- 网络结构
    - `input_layer:784`，输入层，784个神经元用于接收一张图（28*28）展开的像素
    - `hidden1_layer:16`，隐藏层，16个神经元初步提取基础特征
    - `hidden2_layer:16`，隐藏层，16个神经元提取边缘几何特征
    - `output_layer:10`，输出层，10个神经元综合特征输出结果
- 激活函数
    - 中间隐藏层用`relu`
    - 最后输出层用`softmax`


### 训练结果分析

用60000张图片训练得到的模型，再用10000张测试图片来评估，该模型测试集上预测正确率达：`92.87%`，远超之前的逻辑回归模型结果：`18.97%`，由此初窥深度神经网络的魔力。

- 相应实现源码见代码仓：[https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework](https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework)
- 相关keras使用指导：[https://keras.io/zh/getting-started/sequential-model-guide/](https://keras.io/zh/getting-started/sequential-model-guide/)

## 相关链接

---

> 1. 文科生都能零基础学AI？清华这门免费课程让我信了，[link](https://blog.csdn.net/qq_17256689/article/details/123290351)
> 2. 清华青年AI自强作业hw2：线性回归预测，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 3. 清华青年AI自强作业hw3_1：用线性回归模型拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280024)
> 4. 清华青年AI自强作业hw3_2：前向传播和反向传播实战，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
