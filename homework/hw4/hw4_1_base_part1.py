import cv2        # 导入cv2模块，cv2模块的安装采用指令conda install opencv，安装之后导入cv2模块即可成功
import tensorflow as tf        # 导入tensorflow模块
import pandas as pd        # 导入panda模块，重命名为pd
import random        # 导入random模块
import numpy as np        # 导入numpy模块，重命名为np
import time        # 导入time模块

# train&test file name
trainfile = "./train.csv"        # 训练文件地址
testfile = "./test.csv"        # 测试文件地址

#基本参数设置
batch_size=128        # 一个batch中的样本数量为128
lr=0.1        # 学习率为0.1
epochs=10        # 数据集通过训练模型的次数，也可称为训练次数
seed=220        # 随机种子，设置为220
sample_size=0        # 样本尺寸，初始化为0
imageSize=227*227        # 图片尺寸为227*227=51529

# load image data and label from csv
def loadData(readPath):        # loadData可以在补全中选用（若电脑配置够的话，如16G内存），默认不使用
    imageData=[]        # imageData初始化为空列表
    readData = {0}        # 数据读入 待补充
    imgName = readData['imgName']        # 读入图片名称
    label = readData['label']        # 读入图片标记
    for i in range(len(imgName)):        # 循环，在读入的图片名称中循环
        imageData.append(cv2.imread({0}))        # 读入图片数据，存入imgData 待补充 可借鉴loadImg函数
    imageData = np.array(imageData)        # 将imgData格式修改为array格式
    label = np.array(label)        # 将标记修改为array格式
    return imageData,label        # 返回图片数据和标记
    
#load imagename and label from csv
def loadImgPath(readPath):
    readData = pd.read_csv(readPath)        # 从csv中读入数据
    imgName = readData['imgName']        # 读入图片名称
    label = readData['label']        # 读入图片标记
    imgName = np.array(imgName)        # 将imgData格式修改为array格式
    label = np.array(label)        # 将标记修改为array格式
    return imgName,label        # 返回图片数据和标记

# load image 
def loadImg(imgPath):
    imageData=[]        # imgData初始化为空列表
    for i in range(len(imgPath)):        # 循环，在读入的图片路径中循环
        imageData.append(cv2.imread(".."+ imgPath[i],cv2.IMREAD_GRAYSCALE))        # 读入图片数据，存入imgData
    imageData = np.array(imageData)# 将imgData格式修改为array格式
    return imageData        # 返回图片数据和标记

#预测
def predict(X):
    num = X.shape[0]        # 返回变量X的长度
    result = []        # 结果初始化为空列表
    for i in range(num):        # 在num中循环
        if X[i]>0.5:        # 判断，如果X的分量X[i]大于0.5
            result.append(1.0)        # 结果，返回分量1.0
        else:        # 否则，即X的分量X[i]小于0.5
            result.append(0.0)        # 结果，返回分量0.0
    return result        # 返回结果result

# generate minibatch
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):        # 随机最小训练集分类函数
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
    mini_batches = []        # 初始化，将mini_batches初始化为空列表
    np.random.seed(seed)        # 设定随机种子

    # Step 1: Shuffle (X, Y)    打乱X和Y的顺序
    permutation = list(np.random.permutation(m))        # 采用permutation函数打乱读入样本的顺序
    shuffled_X = X[permutation, ]        # 采用permutation函数打乱X的顺序
    shuffled_Y = Y[permutation, ]        # 采用permutation函数打乱Y的顺序

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(np.floor(m / mini_batch_size))    # 计算minibatch的分组数目
    for k in range(0, num_complete_minibatches):        # 循环，从0到生成的样本数目
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]        # 生成最小样本集 X
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]        # 生成最小标记集 Y
        mini_batch = (mini_batch_X, mini_batch_Y)        # 组合X和Y
        mini_batches.append(mini_batch)        # 将mini_batch加入mini_batches

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:        # 如果样本数目无法整除最小样本集数目
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, ]        # 生成最小样本集 X
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, ]        # 生成最小样本集 Y
        mini_batch = (mini_batch_X, mini_batch_Y)        # 组合X和Y
        mini_batches.append(mini_batch)        # 将mini_batch加入mini_batches

    return mini_batches        # 返回mini_batches

# 初始化各个层的权重，初始化方法：标准正太分布
def _initializer1(shape, dtype=tf.float32, partition_info=None):        # 初始化程序
    return tf.random_normal(shape)        # 返回随机数

# 下载数据
trainImgP,trainlabel = loadImgPath(trainfile)        # 导入训练数据与标注
testImgP,testlabel = {0}        # 导入测试数据与标注
sample_size=trainImgP.shape[0]        # 初始化样本数目

# 模型的输入和输出
# 为模型的输入输出申请占位符，作为外部数据与网络模型的交互接口
x  = tf.placeholder(tf.float32, shape=[None, imageSize])  #申请占位符 输入图像 N*227*227的矩阵
y_ = tf.placeholder(tf.float32, shape=[None, ])  #申请占位符 输入label N*1的矩阵


# 将tensor图中的输入和变量进行计算  通过tf.layers.dense搭建全连接网络层，并为该层设置对应的输入、神经元个数、激活函数
# 通过units设置神经元的个数，通过activation设置激活函数，可设定的激活函数，请参考https://tensorflow.google.cn/api_docs/python/tf/nn/softmax
A1 = tf.layers.dense(inputs={0},units=5,activation={0},kernel_initializer=_initializer1)  #(0)为待补充,
A2 = tf.layers.dense(inputs={0},units=3,activation={0},kernel_initializer={0})  #(0)为待补充
y  = tf.layers.dense(inputs={0},units=1,activation=None)  #(0)为待补充

# 交叉熵 用来度量y_与y之间的差异性
y=tf.reshape(y,[-1,])        # 调整矩形y的形状
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels={0}, logits={0})        # 交叉熵函数

#设置learning rate decay策略，随着迭代次数的增加，学习率成指数逐渐减小，减小公式为：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
global_step = tf.Variable(0)        # 初始化
# 采用tf.train.exponential_decay函数设置学习率
learning_rate = tf.train.exponential_decay({0},global_step,decay_steps=sample_size/batch_size,decay_rate={0},staircase=True)

# 训练 利用梯度下降法，以0.01的学习率最小化目标函数（cross_entropy）
train_step = tf.train.AdamOptimizer({0}).minimize({0},global_step=global_step)
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# 创建Session，用于启动tensor图
sess = tf.InteractiveSession()

# 调用global_variables_initializer函数，将前面定义的Variable变量按照设置的初始化方式，进行初始化
sess.run(tf.global_variables_initializer())

# 在测试集上，测试当前模型的精度
def Ttest(testImgPath,testL):        # 测试函数
    newTestImg = loadImg({0}).reshape(-1,imageSize)        # 读入图片并采用reshape函数调整形状
    [test_pred] = sess.run([y], feed_dict={x: {0}, y_: {0}})        # 启动sess，计算test_pred
    test_pred1 = predict(test_pred)        # 利用predict函数计算预测值
    acc = np.mean(test_pred1 == testlabel)*100        # 采用np.mean计算均值
    sumPred = np.sum(test_pred1)        # 计算总和
    return acc, sumPred        # 返回均值与和

# 在训练集上，测试当前模型的精度，考虑到大家CPU的内存容量有限，在测试时，是按照minibatch的方式进行
def Ttrain(trainImagePath,trainL,seed):        # 训练函数
    trainYesNum = 0        # 初始化trainYesNum为0
    sumPredNum=0        # 初始化sumPredNum为0
    minibatches = random_mini_batches(trainImagePath, trainL, batch_size, seed)        # 采用random_mini_batchesh函数计算minibatches
    for minibatchT in minibatches:        # 在minibatches中循环
        (trainTX,trainTY)=minibatchT        # 组合trainTX与trainTY为minibatchT
        trainTX=loadImg(trainTX).reshape(-1,imageSize)        # 采用loadImg函数读入图片，存入trainTX
        [trainPred]=sess.run([y], feed_dict={x: trainTX, y_: trainTY})        # 计算trainPred
        train_pred1 = predict(trainPred)        # 利用predict函数计算预测值
        sumPredNum+=np.sum(train_pred1)        # 采用np.sum计算和
        trainYesNum+=np.sum(train_pred1 == trainTY)        # 采用np.sum计算和
    acc=trainYesNum*1.0*100/(len(minibatches)*batch_size)        # 计算acc
    return acc, sumPredNum        # 返回acc与sumPredNum
    
epoch_cost = 0        # 初始化epoch_cost为0
for i in range(epochs):        # 在0至epochs中循环
    start = time.clock()        # 将当前时间计入start
    epoch_cost+=1        # epoch_cost加1
    seed = seed + 1        # 改变seed，自增1
    minibatches = random_mini_batches(trainImgP, trainlabel, batch_size, seed)        # 利用random_mini_batches计算minbatches
    
    for j in range(len(minibatches)):        # 在minibatches中循环
        print(j)          #为了查看程序是否还活着，建议保留此项输出
        minibatch = minibatches[j]        # 选出一个minibatch
        (minibatch_X, minibatch_Y) = minibatch        # 将minibatch的图片名称和标注数据导出
        minibatch_X = loadImg(minibatch_X)        # 读入图片
        newX = minibatch_X.reshape(-1,imageSize)        # 重整图片形状
        train_step.run(feed_dict={x: {0}, y_: {0}})
        
    end = time.clock()        # 将当前时间存入end
    print('epochs:{0} takes time :{1} s'.format(epoch_cost,(end-start)))        # 打印结果

logName = "./log1.txt"        # 初始化日志名称为log1
pf = open(logName, "w")        # 打开pf文件

print('epochs:{0},minibantches:{1}'.format(int(epoch_cost), int(batch_size)))        # 打印训练信息
pf.write('epochs:{0},minibantches:{1}\n'.format(int(epoch_cost), int(batch_size)))        # 输出训练信息

# 保存当前模型的学习率lr、在minibatch上的测试精度，并打印
[cross_entropy_loss,y_pred,lr_out] = sess.run([loss,{0},learning_rate],feed_dict={x: newX, y_: minibatch_Y})
y_pred1 = predict(y_pred)
print('loss:{:f},lr:{:f},minibatch Accuracy: {:f}'.format(cross_entropy_loss[0],lr_out, (np.mean(y_pred1 == minibatch_Y) * 100)))
pf.write('loss:{:f},lr:{:f},minibatch Accuracy: {:f}\n'.format(cross_entropy_loss[0],lr_out, (np.mean(y_pred1 == minibatch_Y) * 100)))

# 保存当前模型在test上的测试精度，并打印保存信息
acc,sumPred = Ttest(testImgP,testlabel)
print('test predict sum is: {:f},test Accuracy: {:f}'.format(sumPred,acc))
pf.write('test predict sum is: {:f},test Accuracy: {:f}\n'.format(sumPred,acc))

# 保存当前模型在train上的测试精度，并打印保存信息
acc,sumPred = Ttrain(trainImgP,trainlabel,seed)
print('train predict sum is: {:f},train Accuracy: {:f}'.format(sumPred,acc))
pf.write('train predict sum is: {:f},train Accuracy: {:f}\n\n'.format(sumPred,acc))

pf.close()        # 关闭文件



