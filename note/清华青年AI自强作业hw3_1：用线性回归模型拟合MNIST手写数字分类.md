1. @[toc](清华青年AI自强作业hw3_1：用线性回归模型拟合MNIST手写数字分类)

    ![在这里插入图片描述](https://img-blog.csdnimg.cn/c3c2e4b7dae44bef909a63d75eeb227f.jpeg)

    > 一起学AI系列博客：[目录索引](https://blog.csdn.net/qq_17256689/article/details/130910780)

    hw3_1：用线性回归模型拟合MNIST手写数字分类

    - 初步体验Tensorflow编程环境
    - 体会用回归模型网络拟合效果
    
    

    ## 实现过程

    ----

    尝试后发现hw3_1/hw3_3的参考代码为TF1.x框架代码，升级到TF2.x框架多为不便（[升级踩坑记录](https://blog.csdn.net/qq_17256689/article/details/131198032)），于是采用TF2.x中的keras框架重新写了一遍。
    
    
    
    ### 思路分析
    
    1. 先学习参考代码的框架思路
    2. 研究MNIST的Keras训练流程
    3. 进行改造和适配完成本次任务
    
    
    
    ### 逻辑回归二分类模型
    
    1. 核心思路
        1. 用逻辑回归的二分类模型分别训练十个模型，分别处理0-9的分类
    
    2. MNIST数据预处理(load)
        1. 读取MNIST数据，调整处理成网络能接受的batch切分，训练集、测试集划分，标签匹配
    
    3. 模型训练过程(train)
        1. 模型搭建
            1. 模型为WX+B，第一层为输入层(784个神经元)，第二层为输出层(1个神经元)，二分类逻辑回归
            2. 单个类别神经网络层数设置、学习率、梯度下降方法、loss设置
            3. 保存每个batch的model参数，并获取测试集上的最新识别率，此时不开启反向传播更新参数
    
        2. 模型训练
            1. 前向推理与模型参数更新
            2. 按规律预测测试集的精度
    
    4. 模型推理过程(predict)
        1. 在测试集上验证预测精度
    
    5. 关键点
        1. 涉及到数据标签的one-hot编码理解
    
    
    
    
    ### 训练结果分析
    
    从训练结果来看，用60000张图片训练图片得到的模型，再用10000张测试图片来评估，该模型预测正确率仅：`18.97%`，可见此模型设计是不合理的，太过简单，不能很好地表征该复杂任务。下一篇博客将对此模型进行改进，加深网络，便会得到很好的效果。
    
    
    
    相应实现源码见代码仓：[https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework](https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework)。
    
    
    
    **参考资料**
    
    1. [https://github.com/AnkushMalaker/TF2-MNIST](https://github.com/AnkushMalaker/TF2-MNIST)
    
    1. MNIST classification with TF2.0 Keras，[link](https://www.kaggle.com/code/hrideshkohli/mnist-classification-with-tf2-0-keras/notebook)
    2. TF2.0的安装与MNIST手写识别的实现，[link](https://zhuanlan.zhihu.com/p/85147895)
    3. 手把手教程：深度学习入门项目MNIST手写数字分类任务，[Pytorch实现](https://blog.csdn.net/qq_17256689/article/details/123145952)
    
    
    
    ## 相关链接
    
    ---
    
    > 1. 文科生都能零基础学AI？清华这门免费课程让我信了，[link](https://blog.csdn.net/qq_17256689/article/details/123290351)
    > 2. 清华青年AI自强作业2：线性回归预测，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
    > 2. 清华青年AI自强作业hw3_2：前向传播和反向传播实战，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
