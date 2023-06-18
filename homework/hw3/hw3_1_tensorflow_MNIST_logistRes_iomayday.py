
#以下函数的使用方法，请参考：https://tensorflow.google.cn/api_docs/python/
# keras使用参考： https://keras.io/zh/getting-started/sequential-model-guide/

import tensorflow as tf # 导入 tensorflow 库，并且重名为 tf, 便于后面的简写 tf
import numpy as np  # 导入 numpy 库，并且重名为 np, 便于后面的简写 np


#基本参数设置
FLATEN_NERUAL_NUM = 784 # =28*28
CLASS_MODE = 10  # 数据共10种分类
NUM_CLASSES = 2 # 二分类问题
BATCH_SIZE = 50 # 样本大小50
EPOCHS = 10 # 一次训练轮数


# 加载数据集，建议提前到官网上下载MNIST数据集，并解压到./MNIST文件夹下
# MNIST下载地址：http://yann.lecun.com/exdb/mnist/
def loadData(): # 定义一个 loadData 函数
    mnist = tf.keras.datasets.mnist.load_data() # mnist: (x_train, y_train),(x_test, y_test)
    return mnist # 返回读取的数据 mnist


# 对模型输出的结果进行评判，>0.5为“正”，<0.5为“负”
def judge_model(X, Y):   # 定义一个函数 predict， 作用是用来进行预测
    num = X.shape[0]  # 通过 shape 属性，得到 X 行的个数
    result = [] # 定义一个空的列表 result ，后面通过 append 的方式，向里面添加元素
    right_num = 0
    for i in range(num):  # for循环语句， i 从0，1，2, 到 num -1
        if X[i]>0.5: # 如果 X[i] 大于 0.5
            result.append(1.0) # 将 1.0 添加到列表 result 中
        else: # 否则，X[i] 小于或等于 0.5
            result.append(0.0)  # 将 0.0 添加到列表 result 中
        if (Y[i] == 1  and X[i] > 0.5):
            right_num += 1

    # print("acc:{0:.2f}%".format(right_num / num * 100))
    return result, right_num # 返回 result 的结果


# 二分类问题one-hot
def one_hot_encoding(y, Num):
    for i in range(y.shape[0]):
        if y[i] == Num:
            y[i] = 1
        else :
            y[i] = 0  


# 模型搭建、训练、存储
def model(mnist, Num): # 定义一个  model 函数
    (x_train, y_train), (x_test, y_test) = mnist
    x_test,x_train = x_test.reshape((10000,28,28,1)),x_train.reshape(60000,28,28,1)
    x_test,x_train = x_test/255.0,x_train/255.0

    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), # 将输入层样本铺平展开28*28=>784
        tf.keras.layers.Dense(1, activation="sigmoid", input_dim=FLATEN_NERUAL_NUM), # WX+B逻辑回归模型
    ])
    # 二分类问题
    keras_model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    one_hot_encoding(y_train, Num)
    one_hot_encoding(y_test, Num)

    # 训练模型
    keras_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = keras_model.evaluate(x_test, y_test)
    print(test_loss)
    print(test_acc)
    keras_model.save('model_bianry_num{}.h5'.format(Num))
    return


# 循环训练每个类别与其他类别的二分类器，保存10个分类器模型
def train(mnist, classNum):
    for i in range(classNum): # for 循环语句， 遍历所有 classNum的类别， 
        model(mnist, i)   


def predict(mnist, classNum):
    # load test data
    (x_train, y_train), (x_test, y_test) = mnist
    x_test,x_train = x_test.reshape((10000,28,28,1)),x_train.reshape(60000,28,28,1)
    x_test,x_train = x_test/255.0,x_train/255.0
    ok_num = 0

    # evaluate test data
    for num in range(classNum): # for 循环语句， 遍历所有 classNum的类别
        my_model = tf.keras.models.load_model('model_bianry_num{}.h5'.format(num))
        pred = my_model.predict(x_test[:])
        one_hot_encoding(y_test, num)
        result, right_num = judge_model(pred, y_test)
        ok_num += right_num
        # pred_result == y_test
    print("acc:{0:.2f}%".format(ok_num / y_test.shape[0] * 100))


# 模型测试
def test_model():
    mnist = loadData() # 调用 loadData 函数， 导入数据 
    classNum = CLASS_MODE # 类别 初始化赋值为 10 ， 共有 10 类
    
    # 训练
    train(mnist, classNum)

    # 推理
    predict(mnist, classNum)


if __name__ == '__main__': # 主程序
    test_model() # 调用 test_model 函数 


    
