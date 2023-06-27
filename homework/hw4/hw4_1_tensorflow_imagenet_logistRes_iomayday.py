
#以下函数的使用方法，请参考：https://tensorflow.google.cn/api_docs/python/
# keras使用参考： https://keras.io/zh/getting-started/sequential-model-guide/

import cv2  # 导入cv2模块，cv2模块的安装采用指令conda install opencv，安装之后导入cv2模块即可成功
import tensorflow as tf  # 导入tensorflow模块
import pandas as pd  # 导入panda模块，重命名为pd
import numpy as np  # 导入numpy模块，重命名为np
import time  # 导入time模块

# train&test file name
trainfile = "./train.csv"  # 训练文件地址
testfile = "./test.csv"  # 测试文件地址

# 基本参数设置
BATCH_SIZE = 10 # 样本大小50
EPOCHS = 10 # 一次训练轮数
MODEL_NAME = 'model_bianry_dog_logres.h5'
batch_size = 10  # 一个batch中的样本数量为128
lr = 0.1  # 学习率为0.1
decay_rate = 0.8     # 学习率衰为0.8
epochs = 10  # 数据集通过训练模型的次数，也可称为训练次数
sample_size = 1000  # 总体样本抽样个数
imageSize = 227 * 227  # 图片尺寸为227*227=51529


# load image data and label from csv
def loadData(readPath):  # loadData可以在补全中选用（若电脑配置够的话，如16G内存），默认不使用
    imageData = []  # imageData初始化为空列表
    readData = pd.read_csv(readPath)  # 数据读入 待补充
    imgName = readData['imgName']  # 读入图片名称
    label = readData['label']  # 读入图片标记
    for i in range(len(imgName)):  # 循环，在读入的图片名称中循环
        imageData.append(cv2.imread("." + imgName[i], cv2.IMREAD_GRAYSCALE))  # 读入图片数据，存入imgData 待补充 可借鉴loadImg函数
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


# load image
def loadImg(imgPath):
    imageData = []  # imgData初始化为空列表
    for i in range(len(imgPath)):  # 循环，在读入的图片路径中循环
        imageData.append(cv2.imread(".." + imgPath[i], cv2.IMREAD_GRAYSCALE))  # 读入图片数据，存入imgData
    imageData = np.array(imageData)  # 将imgData格式修改为array格式
    return imageData  # 返回图片数据和标记


# 搭建网络结构
def net():
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), # 将输入层样本铺平展开
        tf.keras.layers.Dense(5,
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation="relu", input_dim=imageSize),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3,
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                              bias_initializer=tf.keras.initializers.Zeros(),
                              activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return nn


# 训练一个二分类器确定是否为狗狗
def train(trainImgP, trainlabel):
    total_start = time.perf_counter()  # 将当前时间写入total_start
    # 搭建网络模型
    keras_model = net()
    # 二分类问题
    # sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, decay=decay_rate, nesterov=False)
    adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    keras_model.compile(optimizer=adam,
                loss='binary_crossentropy',
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
    x = loadImg(x) # jpeg data
    x = x.reshape(-1, imageSize)  # 重整图片形状
    return x, y


# 模型测试
def test_model():
    # 下载数据
    trainImgP, trainlabel = loadImgPath(trainfile)  # 导入训练数据与标注, path
    testImgP, testlabel = loadImgPath(testfile)  # 导入测试数据与标注

    # 图片形状调整
    trainImgpData, trainlabel = shape_size(trainImgP, trainlabel, sample_size)
    testImgpData, testlabel = shape_size(testImgP, testlabel, sample_size)

    # 训练
    train(trainImgpData, trainlabel)

    # 推理
    predict(testImgpData, testlabel)
    return


if __name__ == '__main__': # 主程序
    test_model() # 调用 test_model 函数 

