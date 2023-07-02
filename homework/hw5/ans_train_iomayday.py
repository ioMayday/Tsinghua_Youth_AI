# coding=utf-8

#以下函数的使用方法，请参考：https://tensorflow.google.cn/api_docs/python/
# keras使用参考： https://keras.io/zh/getting-started/sequential-model-guide/

# 模块导入
import cv2 as cv  # 导入cv2模块，cv2模块的安装采用指令conda install opencv，安装之后导入cv2模块即可成功
import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块，重命名为np
import time  # 导入time模块
from ans_data_iomayday import get_data_set

# 支持GPU和本地调参设置
GPU_USE = 1
DEBUG_MODE = 0

if GPU_USE == 1:
  # GPU服务器相关设置
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 开启GPU加速


# 基本训练超参设置
MODEL_NAME = 'model_cifar10_cnn.h5'
lr = 0.1  # 学习率为0.1
decay_rate = 0.001     # 学习率衰减
image_input_shape = []


if DEBUG_MODE == 1:
  # debug
  BATCH_SIZE = 10 # 样本大小50
  EPOCHS = 10 # 一次训练轮数
  sample_size = 1000  # 总体样本抽样个数
else:
  # release
  BATCH_SIZE = 128
  EPOCHS = 50 # 一次训练轮数


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


# 搭建网络结构
def net():
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, # 特征个数
                               (3, 3), # 卷积核大小
                               padding='same',  # 32*32 -> 32*32
                               input_shape=image_input_shape,
                               activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(24,
                               (3, 3),
                               padding='valid', # 16*16 -> 14*14
                               activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(), # 将样本铺平展开24*(7*7)=1176
        tf.keras.layers.Dense(128,
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128,
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10,
                              activation='softmax'),
    ])
    return nn


# 训练一个十分类器
def train(trainImgP, trainlabel):
    total_start = time.perf_counter()  # 将当前时间写入total_start
    # 搭建网络模型
    keras_model = net()
    # 二分类问题
    # sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, decay=decay_rate, nesterov=False)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    # opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
    keras_model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # 训练模型
    keras_model.fit(trainImgP, trainlabel, epochs=EPOCHS, batch_size=BATCH_SIZE) # trainlabel已经是one-hot
    # 评估模型
    test_loss, test_acc = keras_model.evaluate(trainImgP, trainlabel) # 评估最终精度
    print(test_loss, test_acc)
    # 保存模型
    keras_model.save(MODEL_NAME)
    total_end = time.perf_counter()  # 将当前时间放入total_end中
    time_duration = total_end - total_start
    str_time_epochs = 'total epochs:{0} takes time :{1:.2f} s'.format(EPOCHS, time_duration)
    print(str_time_epochs)  # 打印结果
    return


def predict(testImgP, testlabel):
    my_model = tf.keras.models.load_model(MODEL_NAME)
    test_loss, test_acc = my_model.evaluate(testImgP, testlabel) # 评估最终精度
    print(test_loss, test_acc)


def shape_size(x, y, sample_size):
    if (sample_size > x.shape[0]):
        sample_size = x.shape[0]
    y = y[0:sample_size]
    x = x[0:sample_size]
    return x, y


# 模型测试
def test_model():
    # 下载数据
    train_x, train_y, test_x, test_y = loadData()  # 调用loadData()函数，返回训练与测试数据集
    print('x_train shape:', train_x.shape)
    print(train_x.shape[0], 'train samples')
    print(test_x.shape[0], 'test samples')    
    train_x /= 255
    test_x /= 255

    _IMAGE_SIZE = train_x.shape[0]  # train_x的第一个维度赋值给_IMAGE_SIZE，样本的总个数
    print(_IMAGE_SIZE)
    _IMAGE_SIZE = test_x.shape[0]  # train_x的第一个维度赋值给_IMAGE_SIZE，样本的总个数
    print(_IMAGE_SIZE)
    global image_input_shape
    image_input_shape = train_x.shape[1:] # 单个样本的维度大小，如一张RGB图
    print(image_input_shape)

    if DEBUG_MODE == 1:
      # 开发调试缩减数据量
      train_x, train_y = shape_size(train_x, train_y, sample_size)
      # test_x, test_y = shape_size(test_x, test_y, sample_size)

    # 训练
    train(train_x, train_y)

    # 推理
    predict(test_x, test_y)
    return


if __name__ == '__main__': # 主程序
    test_model() # 调用 test_model 函数 
