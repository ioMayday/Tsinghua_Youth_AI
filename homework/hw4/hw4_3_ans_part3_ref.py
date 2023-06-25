import cv2  # 导入cv2模块，cv2模块的安装采用指令conda install opencv，安装之后导入cv2模块即可成功
import tensorflow as tf  # 导入tensorflow模块
import pandas as pd  # 导入panda模块，重命名为pd
import random  # 导入random模块
import numpy as np  # 导入numpy模块，重命名为np
import time  # 导入time模块

# train&test file name
trainfile = "./train.csv"  # 训练文件地址
testfile = "./test.csv"  # 测试文件地址

# 基本参数设置
batch_size = 128  # 一个batch中的样本数量为128
lr = 0.1  # 学习率为0.1
epochs = 10  # 数据集通过训练模型的次数，也可称为训练次数
seed = 220  # 随机种子，设置为220
sample_size = 0  # 样本尺寸，初始化为0
imageSize = 227 * 227  # 图片尺寸为227*227=51529


# load imagename and label from csv    loadData函数可以在补全中选用（若电脑配置够的话，如16G内存），默认不使用。
def loadData(readPath):
    imageData = []  # imageData初始化为空列表
    readData = pd.read_csv(readPath)  # 数据读入
    imgName = readData['imgName']  # 读入图片名称
    label = readData['label']  # 读入图片标记
    for i in range(len(imgName)):  # 循环，在读入的图片名称中循环
        imageData.append(cv2.imread("." + imgName[i], cv2.IMREAD_GRAYSCALE))  # 读入图片数据，存入imgData
    imageData = np.array(imageData)  # 将imgData格式修改为array格式
    label = np.array(label)  # 将标记修改为array格式
    return imageData, label  # 返回图片数据和标记


# load imagename and label from csv
def loadImgPath(readPath):
    readData = pd.read_csv(readPath)  # 从csv中读入数据
    imgName = readData['imgName']  # 读入图片名称
    label = readData['label']  # 读入图片标记
    imgName = np.array(imgName)  # 将imgData格式修改为array格式
    label = np.array(label)  # 将标记修改为array格式
    return imgName, label  # 返回图片数据和标记


def loadImg(imgPath):
    imageData = []  # imgData初始化为空列表
    for i in range(len(imgPath)):  # 循环，在读入的图片路径中循环
        imageData.append(cv2.imread(".." + imgPath[i],
                                    cv2.IMREAD_GRAYSCALE))  # 读入图片数据，存入imgData，需要调用函数cv2.imread，采用cv2.IMREAD_GRAYSCALE关键字
    imageData = np.array(imageData)  # 将imgData格式修改为array格式
    return imageData  # 返回图片数据和标记


# 预测
def predict(X):
    num = X.shape[0]  # 返回变量X的长度
    result = []  # 结果初始化为空列表
    for i in range(num):  # 在num中循环
        if X[i] > 0.5:  # 判断，需要选用合适的阈值
            result.append(1.0)  # 结果，返回分量1.0
        else:  # 否则，即X的分量X[i]小于0.5
            result.append(0.0)  # 结果，返回分量0.0
    return result  # 返回结果result


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):  # 随机最小训练集分类函数
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape ( number of examples,input size)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape ( number of examples, 1)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]  # number of training examples
    mini_batches = []  # 初始化，将mini_batches初始化为空列表
    np.random.seed(seed)  # 设定随机种子

    # Step 1: Shuffle (X, Y)    打乱X和Y的顺序
    permutation = list(np.random.permutation(m))  # 采用permutation函数打乱读入样本的顺序
    shuffled_X = X[permutation,]  # 采用permutation函数打乱X的顺序
    shuffled_Y = Y[permutation,]  # 采用permutation函数打乱Y的顺序

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))  # 计算minibatch的分组数目
    for k in range(0, num_complete_minibatches):  # 循环，从0到生成的样本数目
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]  # 生成最小样本集 X
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]  # 生成最小标记集 Y
        mini_batch = (mini_batch_X, mini_batch_Y)  # 组合X和Y
        mini_batches.append(mini_batch)  # 将mini_batch加入mini_batches

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:  # 如果样本数目无法整除最小样本集数目
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, ]  # 生成最小样本集 X
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, ]  # 生成最小样本集 Y
        mini_batch = (mini_batch_X, mini_batch_Y)  # 组合X和Y
        mini_batches.append(mini_batch)  # 将mini_batch加入mini_batches

    return mini_batches  # 返回mini_batches


def _initializer1(shape, dtype=tf.float32, partition_info=None):  # 初始化程序
    return tf.random_normal(shape)  # 返回随机数


def _initializer2(shape, dtype=tf.float32, partition_info=None):  # 初始化程序
    initializer = tf.contrib.layers.xavier_initializer()  # 采用函数 tf.contrib.layers.xavier_initializer()
    return initializer(shape)  # 返回函数调用值


def _regularizer1(var):  # 正则函数
    return tf.contrib.layers.l2_regularizer(0.01)(var)  # l2正则


def fully_connected_batchNorm(prev_layer, num_units, is_training):  # 全连接层函数
    """
   :param prev_layer: Tensor
        该层神经元输入
    :param num_units: int
        该层神经元结点个数
    :param is_training: bool or Tensor
        表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
    :returns Tensor
        一个新的全连接神经网络层
    """
    layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None,
                            kernel_initializer=_initializer2)  # 创建预训练层
    layer = tf.layers.batch_normalization(layer, training=is_training)  # 归一化
    layer = tf.nn.relu(layer)  # 选用激活函数layer
    return layer  # 返回layer


# 下载数据
trainImgP, trainlabel = loadImgPath(trainfile)  # 下载训练数据与标记
testImgP, testlabel = loadImgPath(testfile)  # 下载测试数据与标记
sample_size = trainImgP.shape[0]  # 初始化样本数目

# 模型的输入和输出
# 为模型的输入输出申请占位符，作为外部数据与网络模型的交互接口
x = tf.placeholder(tf.float32, shape=[None, imageSize])  # 申请占位符 输入图像 N*227*227的矩阵
y_ = tf.placeholder(tf.float32, shape=[None, ])  # 申请占位符 输入label N*1的矩阵

'''
# 将tensor图中的输入和变量进行计算  通过tf.layers.dense搭建全连接网络层，并为该层设置对应的输入、神经元个数、激活函数
# 通过units设置神经元的个数，通过activation设置激活函数，可设定的激活函数，请参考https://tensorflow.google.cn/api_docs/python/tf/nn/softmax
#,kernel_initializer=tf.zeros()
A1 = tf.layers.dense(inputs=x, units=1024,activation=tf.nn.elu,kernel_initializer=_initializer2)  #(0)为待补充,
A2 = tf.layers.dense(inputs=A1,units=2048,activation=tf.nn.elu,kernel_initializer=_initializer2)  #(0)为待补充
A3 = tf.layers.dense(inputs=A2,units=2048,activation=tf.nn.elu,kernel_initializer=_initializer2)  #(0)为待补充
A4 = tf.layers.dense(inputs=A3,units=4096,activation=tf.nn.elu,kernel_initializer=_initializer2)  #(0)为待补充
y  = tf.layers.dense(inputs=A9,units=1, activation=None)  #(0)为待补充'''

# 增加全连接层，创建网络
A1 = fully_connected_batchNorm(prev_layer=x, num_units=1024, is_training=True)  # (0)为待补充,
A2 = fully_connected_batchNorm(prev_layer=A1, num_units=2048, is_training=True)  # (0)为待补充
A3 = fully_connected_batchNorm(prev_layer=A2, num_units=2048, is_training=True)  # (0)为待补充
A4 = fully_connected_batchNorm(prev_layer=A3, num_units=1024, is_training=True)  # (0)为待补充
y = tf.layers.dense(inputs=A4, units=1, activation=None)

# 交叉熵 用来度量y_与y之间的差异性
y = tf.reshape(y, [-1, ])
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)  # 交叉熵函数
loss = cross_entropy  # + tf.losses.get_regularization_loss()

# 设置learning rate decay策略，随着迭代次数的增加，学习率成指数逐渐减小，减小公式为：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
global_step = tf.Variable(0)  #
learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=sample_size / batch_size, decay_rate=0.8,
                                           staircase=True)

# 训练 利用梯度下降法，以0.01的学习率最小化目标函数（cross_entropy）
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# 创建Session，用于启动tensor图
sess = tf.InteractiveSession()

# 调用global_variables_initializer函数，将前面定义的Variable变量按照设置的初始化方式，进行初始化
sess.run(tf.global_variables_initializer())


def Ttest(testImgPath, testL):  # 测试函数
    newTestImg = loadImg(testImgPath).reshape(-1, imageSize)  # 读入图片并采用reshape函数调整形状
    [test_pred] = sess.run([y], feed_dict={x: newTestImg, y_: testL})  # 启动sess，计算test_pred
    test_pred1 = predict(test_pred)  # 利用predict函数计算预测值
    acc = np.mean(test_pred1 == testlabel) * 100  # 采用np.mean计算均值
    sumPred = np.sum(test_pred1)  # 计算总和
    return acc, sumPred  # 返回均值与和


def Ttrain(trainImagePath, trainL, seed):  # 训练函数
    trainYesNum = 0  # 初始化trainYesNum为0
    sumPredNum = 0  # 初始化sumPredNum为0
    minibatches = random_mini_batches(trainImagePath, trainL, batch_size, seed)  # 采用random_mini_batchesh函数计算minibatches
    for minibatchT in minibatches:  # 在minibatches中循环
        (trainTX, trainTY) = minibatchT  # 组合trainTX与trainTY为minibatchT
        trainTX = loadImg(trainTX).reshape(-1, imageSize)  # 采用loadImg函数读入图片，存入trainTX
        [trainPred] = sess.run([y], feed_dict={x: trainTX, y_: trainTY})  # 计算trainPred
        train_pred1 = predict(trainPred)  # 利用predict函数计算预测值
        sumPredNum += np.sum(train_pred1)  # 采用np.sum计算和
        trainYesNum += np.sum(train_pred1 == trainTY)  # 采用np.sum计算和
    acc = trainYesNum * 1.0 * 100 / (len(minibatches) * batch_size)  # 计算acc
    return acc, sumPredNum  # 返回acc与sumPredNum


epoch_cost = 0  # 初始化epoch_cost为0
logName = "./log3.txt"  # 日志名称，命名为log3-3.txt
pf = open(logName, "w")  # 创建文件pf，并以读写模式打开
total_start = time.clock()  # 将当前时间写入total_start
for i in range(epochs):  # 循环遍历epochs
    start = time.clock()  # 将当前时间写入start
    epoch_cost += 1  # epoch_cost自增1
    seed = seed + 1  # 随机种子自增1
    minibatches = random_mini_batches(trainImgP, trainlabel, batch_size, seed)  # 采用random_mini_batches函数初始化minibatches

    for j in range(len(minibatches)):  # 循环遍历minibatches
        print(j)
        minibatch = minibatches[j]  # 将minibatches[j]的值赋给minibatch
        (minibatch_X, minibatch_Y) = minibatch  # 导出minibatch的图片地址和标注
        minibatch_X = loadImg(minibatch_X)  # 加载图片
        # print(minibatch_X.shape)
        newX = minibatch_X.reshape(-1, imageSize)  # 重整图片尺寸
        train_step.run(feed_dict={x: newX, y_: minibatch_Y})  # 将数据放入训练网络

    end = time.clock()  # 将当前时间放入end变量
    print('epochs:{0} takes time :{1} s'.format(epoch_cost, (end - start)))  # 打印结果
    pf.write('epochs:{0} takes time :{1} s\n'.format(epoch_cost, (end - start)))  # 将结果输出到pf文件中

print('epochs:{0},minibantches:{1}'.format(int(epoch_cost), int(batch_size)))  # 打印结果
pf.write('epochs:{0},minibantches:{1}\n'.format(int(epoch_cost), int(batch_size)))  # 将结果输出到pf文件中

# 保存当前模型的学习率lr、在minibatch上的测试精度，打印结果并保存
[cross_entropy_loss, y_pred, lr_out] = sess.run([loss, y, learning_rate], feed_dict={x: newX, y_: minibatch_Y})
y_pred1 = predict(y_pred)

print('loss:{:f},lr:{:f},minibatch Accuracy: {:f}'.format(cross_entropy_loss[0], lr_out,
                                                          (np.mean(y_pred1 == minibatch_Y) * 100)))
pf.write('loss:{:f},lr:{:f},minibatch Accuracy: {:f}\n'.format(cross_entropy_loss[0], lr_out,
                                                               (np.mean(y_pred1 == minibatch_Y) * 100)))

# 保存当前模型在test上的测试精度，打印结果并保存
acc, sumPred = Ttest(testImgP, testlabel)
print('test predict sum is: {:f},test Accuracy: {:f}'.format(sumPred, acc))
pf.write('test predict sum is: {:f},test Accuracy: {:f}\n'.format(sumPred, acc))

# 保存当前模型在train上的测试精度，打印结果并保存
acc, sumPred = Ttrain(trainImgP, trainlabel, seed)
print('train predict sum is: {:f},train Accuracy: {:f}'.format(sumPred, acc))
pf.write('train predict sum is: {:f},train Accuracy: {:f}\n\n'.format(sumPred, acc))

total_end = time.clock()  # 将当前时间放入total_end中
print('total epochs:{0} takes time :{1} s'.format(epoch_cost, (total_end - total_start)))  # 打印结果
pf.write('total epochs:{0} takes time :{1} s\n'.format(epoch_cost, (total_end - total_start)))  # 保存结果
pf.close()  # 关闭文件