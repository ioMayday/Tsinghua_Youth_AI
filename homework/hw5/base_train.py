# coding=utf-8
# GPU服务器相关设置
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 模块导入
import numpy as np
import tensorflow as tf
import cv2 as cv
from time import time
from cnn import model_cnn
from data import get_data_set

# 基本训练超参设置
_IMG_SIZE = 32  # 图片大小
_NUM_CHANNELS = 3  # 图像为3通道
_BATCH_SIZE = 256  # 一个batch为256
_CLASS_SIZE = 10  # 10个类别
_ITERATION = 15000  # 设置循环次数，每个为1个mini-batch，调试期间可以调小，训练再调整回来。
_IMAGE_SIZE = 0
_LR = 0.1  # 学习率为0.1
_SEED = 220  # 随机种子

# log 保存地址
_SAVE_MODEL_PATH = "./tensorboard/cifar-10/"  # 先创建好这些文件
_SAVE_BOARD_PATH = "./tensorboard/board/"


# 读取数据
def loadData():
    train_x, train_y = get_data_set(name="train", cifar=10)  # 调用get_data_set，获取训练数据集
    test_x, test_y = get_data_set(name="test", cifar=10)  # 调用get_data_set，获取测试数据集

    # 数据增广 左右翻转
    dataNew = []  # 定义一个数据列表
    labelNew = []  # 定义一个新的标签列表
    for i in range(len(train_x)):  # 遍历整个train_x
        dataNew.append(train_x[i])  # 将第i个train_x加入data_New列表
        dataNew.append(cv.flip(train_x[i], 1))  # 将train_x[i]水平翻转后，加入data_New列表
        labelNew.append(train_y[i])  # 将第train_y[i]加入标签列表
        labelNew.append(train_y[i])  # 将第train_y[i]加入标签列表；因为图像水平翻转后，类别并没有发生变化
    dataNew = np.array(dataNew)  # 数据类型由列表变为numpy的array类型
    labelNew = np.array(labelNew)  # 数据类型由列表变为numpy的array类型
    train_x = dataNew  # 新得到的训练数据集赋值给train_x，达到数据增广的目的
    train_y = labelNew  # 新得到的训练标签数据赋值给train_y
    return train_x, train_y, test_x, test_y  # 返回增广后的训练与测试用到的数据集


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
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
    m = X.shape[0]  # number of training examples # X的第一个维度为训练样本个数
    mini_batches = []  # 定义一个列表
    np.random.seed(seed)  # 保证每次随机产生的数据相同

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))  # 生成一个随机序列，长度为m
    shuffled_X = X[permutation,]  # 得到打乱后的X
    shuffled_Y = Y[permutation,]  # 得到打乱后的Y

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(
        np.floor(m / mini_batch_size))  # 样本总量/mini_batch_size，得到经过多少次mini_batch_size，训练完成一次数据集
    for k in range(0, num_complete_minibatches):  # 遍历num_complete_minibatches次
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]  # 得到一个mini_batch_X
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, ]  # 得到一个mini_batch_Y
        mini_batch = (mini_batch_X, mini_batch_Y)  # 将mini_batch_X、mini_batch_Y打包为一个元组
        mini_batches.append(mini_batch)  # 每一个得到的mini_batch，加入一个列表，最后得到一个数据集的所有mini_batch

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[m - mini_batch_size: m, ]  # 为了保证返回的所有mini-batch 大小均为一致的
        mini_batch_Y = shuffled_Y[m - mini_batch_size: m, ]  # 为了保证返回的所有mini-batch 大小均为一致的
        mini_batch = (mini_batch_X, mini_batch_Y)  # 将mini_batch_X、mini_batch_Y打包为一个元组
        mini_batches.append(mini_batch)  # 得到的mini_batch加入列表
    return mini_batches  # 返回mini_batches列表


# 模型在test数据集上的测试精度
def Ttest(testImage, testL, seed):
    test_accs = []  # 定义一个列表
    minibatches = random_mini_batches(testImage, testL, _BATCH_SIZE, seed)  # 调用random_mini_batches，得到minibatches
    for minibathT in minibatches:  # 遍历minibatches
        (testTX, testTY) = minibathT  # 将元组minibathT，对应赋值给(testTX,testTY)
        test_acc = sess.run(accuracy, feed_dict={x: testTX, y: testTY})  # 运行run函数，得到test_acc
        test_accs.append(test_acc)  # 得到得每一个batch的测试精度加入列表
    sumYesNum = _BATCH_SIZE * np.sum(test_accs)  # 请补全  计算分类正确的图像张数
    acc = sumYesNum * 100 / (len(minibatches) * _BATCH_SIZE)  # 请补全  计算数据集上的精度
    return acc, sumYesNum  # 返回计算得到得精度，与分类正确的图像张数


# 模型在训练集上的测试精度
def Ttrain(trainImage, trainL, seed):
    train_accs = []  # 定义一个列表
    minibatches = random_mini_batches(trainImage, trainL, _BATCH_SIZE, seed)  # 调用random_mini_batches函数，得到所有的batch
    for minibathT in minibatches:  # 遍历minibatches
        (trainTX, trainTY) = minibathT  # 将元组minibathT，对应赋值给(trainTX,trainTY)
        train_acc = sess.run(accuracy, feed_dict={x: trainTX, y: trainTY})  # 调用run函数，传入数据，得到train_acc
        train_accs.append(train_acc)  # # 得到得每一个batch的训练精度加入列表
    sumYesNum = _BATCH_SIZE * np.sum(train_accs)  # 计算分类正确的图像张数
    acc = sumYesNum * 100 / (len(minibatches) * _BATCH_SIZE)  # 计算训练数据集的精度
    return acc, sumYesNum  # 返回计算得到得精度，与分类正确的图像张数


######################## 下载数据 ###############################
train_x, train_y, test_x, test_y = loadData()  # 调用loadData()函数，返回训练与测试数据集
_IMAGE_SIZE = train_x.shape[0]  # train_x的第一个维度赋值给_IMAGE_SIZE

################### 搭建tensorflow流图 ##########################
x, y, output, global_step, y_pred_cls = model_cnn(_BATCH_SIZE)  # 调用model_cnn函数
# 设置learning rate decay策略，
# 随着迭代次数的增加，学习率成指数逐渐减小，减小公式为：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
learning_rate = tf.train.exponential_decay(_LR, global_step, decay_steps=_IMAGE_SIZE / _BATCH_SIZE, decay_rate=0.98,
                                           staircase=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))  # 选择损失函数
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                           global_step=global_step)  # 请补全 构造优化器，采用Adam优化方法进行模型参数优化

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))  # 比较y_pred_cls与y在1轴（列）上的最大值，结果是一个布尔型矩阵
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算分类正确率；tf.cast，把correct_prediction变为float32类型

tf.summary.scalar('lr', learning_rate)  # 用来显示tensorboard中学习率的变化
tf.summary.scalar('loss', loss)  # 用来显示tensorboard中loss的变化
tf.summary.scalar("Accyracy/train", accuracy)  # 用来显示tensorboard中accuracy的变化
tf.summary.histogram('histogram', accuracy)  # 用来显示tensorboard中accuracy变化的直方图
saver = tf.train.Saver(max_to_keep=4)  # 保持最近的4个模型
sess = tf.Session()  # 开启一个会话
merged = tf.summary.merge_all()  # 将summary中的信息merge到一起

train_writer = tf.summary.FileWriter(_SAVE_BOARD_PATH, sess.graph)  # 根据_SAVE_BOARD_PATH路径保存图
sess.run(tf.global_variables_initializer())  # 初始化图中所有变量
#####################################################################

########################### 开始训练 ###############################
total_timeS = time()  # 记录总时间
for i in range(_ITERATION):
    print("正在进行第" + str(i + 1) + "次迭代。")  # 目的在确认程序还在运行
    randidx = np.random.randint(_IMAGE_SIZE, size=_BATCH_SIZE)  # 返回_BATCH_SIZE个随机数,用于mini-batch
    batch_xs = train_x[randidx]  # 得到一个batch数据
    batch_ys = train_y[randidx]  # 得到一个batch数据

    start_time = time()  # 记录mini-batch的时间
    i_global, _ = sess.run([global_step, optimizer],
                           feed_dict={x: batch_xs, y: batch_ys})  # 请补全 调用优化器进行当前mini-batch的训练优化，并返回global_step
    duration = time() - start_time  # 得到训练所用时间，time()为当前时间

    if (i_global % 100 == 0) or (i == _ITERATION - 1):  # 如果i_global是100的整数倍，或者达到最后迭代次数
        _loss, batch_acc = sess.run([loss, accuracy],
                                    feed_dict={x: batch_xs, y: batch_ys})  # 调用run函数，计算损失值loss与每一个batch的acc
        msg = "Glo bal Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
        print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))  # 根据字符串格式化信息，调用format函数，打印相关指标

        resultmerged = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})  # 计算summary中的信息
        train_writer.add_summary(resultmerged, i_global)  # 将计算结果与对应的迭代次数加入tensorboard

#################### 保存当前模型的测试精度 ************************
# 测试集上的精度检测
acc, sumPred = Ttest(test_x, test_y, _SEED)  #
print('test predict sum is: {:f},test Accuracy: {:f}'.format(sumPred, acc))

# 训练集上的精度检测
acc, sumPred = Ttrain(train_x, train_y, _SEED)
print('train predict sum is: {:f},train Accuracy: {:f}'.format(sumPred, acc))  # 输出训练集上训练结果信息

# 保存当前模型
saver.save(sess, save_path=_SAVE_MODEL_PATH, global_step=global_step)
print("Saved checkpoint")

sess.close()  # 关闭会话
print("{0} step cost time :{1} s".format(int(_ITERATION), time() - total_timeS))  # 输出_ITERATION次迭代，总耗费时长，单位为s
