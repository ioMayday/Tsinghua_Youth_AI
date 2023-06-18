
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
        maxIdx = np.argmax(X[i])
        X[i] = 0 * X[i]
        X[i][maxIdx] = 1 #tbd
        if (maxIdx == np.argmax(Y[i])):
            right_num += 1
    print("acc:{0:.2f}%".format(right_num / num * 100))
    return result # 返回 result 的结果


# 模型搭建、训练、存储
def model(mnist): # 定义一个  model 函数
    (x_train, y_train), (x_test, y_test) = mnist
    x_test,x_train = x_test.reshape((10000,28,28,1)),x_train.reshape(60000,28,28,1)
    x_test,x_train = x_test/255.0,x_train/255.0

    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), # 将输入层样本铺平展开28*28=>784
        tf.keras.layers.Dense(16, activation="relu", input_dim=FLATEN_NERUAL_NUM), # hidden_layer1
        tf.keras.layers.Dense(16, activation="relu"), # hidden_layer2
        tf.keras.layers.Dense(10, activation="softmax"), # output_layer
    ])
    # 多分类问题
    keras_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    y_train_one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=CLASS_MODE)
    y_test_one_hot_labels = tf.keras.utils.to_categorical(y_test, num_classes=CLASS_MODE)

    # 训练模型
    keras_model.fit(x_train, y_train_one_hot_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    test_loss, test_acc = keras_model.evaluate(x_test, y_test_one_hot_labels)
    print(test_loss)
    print(test_acc)
    keras_model.save('model_categorical_num.h5')
    return


# 循环训练多分类器模型
def train(mnist): 
    model(mnist)   


def predict(mnist):
    # load test data
    (x_train, y_train), (x_test, y_test) = mnist
    x_test,x_train = x_test.reshape((10000,28,28,1)),x_train.reshape(60000,28,28,1)
    x_test,x_train = x_test/255.0,x_train/255.0
    y_test_one_hot_labels = tf.keras.utils.to_categorical(y_test, num_classes=CLASS_MODE)

    # evaluate test data
    my_model = tf.keras.models.load_model('model_categorical_num.h5')
    pred = my_model.predict(x_test[:])
    result = judge_model(pred, y_test_one_hot_labels)
    # pred_result == y_test


# 模型测试
def test_model():
    mnist = loadData() # 调用 loadData 函数， 导入数据 
    
    # 训练
    train(mnist)

    # 推理
    predict(mnist)


if __name__ == '__main__': # 主程序
    test_model() # 调用 test_model 函数 


    
