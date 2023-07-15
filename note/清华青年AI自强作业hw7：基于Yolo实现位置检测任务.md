@[toc](清华青年AI自强作业hw7：基于Yolo实现位置检测任务)

![在这里插入图片描述](https://img-blog.csdnimg.cn/c3c2e4b7dae44bef909a63d75eeb227f.jpeg)

> 一起学AI系列博客：[目录索引](https://blog.csdn.net/qq_17256689/article/details/130910780)

## 简述

---

hw7作业为基于Yolo模型，对PASCAL_VOC_2007数据集的20类物体进行位置探测。数据集为600张图像，因此建议训练迭代次数小于100次即可。

由于缺乏实验数据，本次作业不进行实战，只对TF1.x版本的参考代码进行思路梳理学习。

- 相应实现源码见代码仓：[https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework](https://github.com/ioMayday/Tsinghua_Youth_AI/tree/master/homework)
- 相关keras使用指导：[https://keras.io/zh/getting-started/sequential-model-guide/](https://keras.io/zh/getting-started/sequential-model-guide/)

## 作业实现

---

- project文件架构

    - ./目录
        - train.py，训练源码，调用utils
        - test.py，测试源码，调用utils
    - testImg目录：测试图像观测结果
    - utils目录：辅助模块代码
        - config.py：配置模型相关参数和文件路径
        - data_pascal_voc.py：读取PASCAL_VOC数据
        - model_yolo.py：网络模型构建
        - timer.py：时间计算
        - download_data.sh：若数据集文件加载较慢，可运行该文件，从网上下载数据集

具体模型构建及层数设计见源码`model_yolo.py`和相关论文，这里提几个特殊的点：

- 模型输出
    - 先划分格子进行分类任务，再根据bounding box进行位置输出
    - 输出label为：1、是否含目标物体；2、目标物体位置；3、目标物体类别
- 训练loss进行合理组合
    - 分类loss（是否含目标）
    - 类别loss（具体哪个目标）
    - 位置loss

基本原理：

![在这里插入图片描述](https://img-blog.csdnimg.cn/6ecb287fac264871882499e25c232f1c.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/5ffcd9fe96ee45b78b69cfb81dd97d26.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/5fd587de84d1469ebf6756c48151fff6.png)



## 相关链接																								

---

> 1. 文科生都能零基础学AI？清华这门免费课程让我信了，[link](https://blog.csdn.net/qq_17256689/article/details/123290351)
> 2. 清华青年AI自强作业hw2：线性回归预测，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 3. 清华青年AI自强作业hw3_1：用线性回归模型拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280024)
> 4. 清华青年AI自强作业hw3_2：前向传播和反向传播实战，[link](https://blog.csdn.net/qq_17256689/article/details/124435599)
> 5. 清华青年AI自强作业hw3_3：用NN网络拟合MNIST手写数字分类，[link](https://blog.csdn.net/qq_17256689/article/details/131280109)
> 6. 清华青年AI自强作业hw4：基于DNN实现狗狗二分类与梯度消失实验，[link](https://blog.csdn.net/qq_17256689/article/details/131424142)
> 7. 清华青年AI自强作业hw5：基于CNN实现CIFAR10分类任务，[link]( https://blog.csdn.net/qq_17256689/article/details/131501974)
> 7. 清华青年AI自强作业hw6：基于ResNet实现IMAGENET分类任务，[link](https://blog.csdn.net/qq_17256689/article/details/131605796)
