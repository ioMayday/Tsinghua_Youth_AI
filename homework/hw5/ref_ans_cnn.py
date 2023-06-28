# coding=utf-8
import tensorflow as tf  # 导入 tensorflow 库，并且重命名为 tf


def print_activations(t):  # 定义一个 print_activations 函数，用于打印出激活函数
    print(t.op.name, '', t.get_shape().as_list)  # get_shape获取一个TensorShape对象，然后通过as_list方法返回每一个维度数


def get_weight(shape):  # 定义一个 get_weight 函数，来得到权重
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, mean=0, stddev=1e-1),
                       name='weights')  # 返回 权重 tf.Variable 变量


def my_conv2d(_input, _filter, _strides, padding='SAME'):  # 定义一个 my_conv2d 函数，构建卷积层
    '''
    #input：输入，4维向量，[batch, in_height, in_width, in_channels]
    #filter：卷积核，4维向量，[filter_height, filter_width, in_channels, out_channels]
             卷积核尺寸必须为奇数
    #Strides：步长，4维向量，针对输入数据的各个维度，stride分别为多少，通常为[1, stride, stride, 1]
    #Padding：边缘扩充，可设置成“SAME”或者“VALID”
    '''
    inputShape = _input.get_shape().as_list()  # get_shape获取一个TensorShape对象，然后通过as_list方法返回每一个维度数
    filterShape = _filter.get_shape().as_list()  # 请补全
    # print(inputShape,filterShape)

    if padding == "SAME":  # 如果 padding 判断等于 SAME
        NewHei = inputShape[1] + (filterShape[0] - 1)  # inputShape 中第二个元素 加上 (filterShape[0]-1) ， 赋值给 NewHei
        NewWid = inputShape[2] + (filterShape[1] - 1)  # 请补全
        offH = int((filterShape[0] - 1) / 2)  # 强制转换为整型，int((filterShape[0]-1)/2) ，并赋值给 offH
        offW = int((filterShape[0] - 1) / 2)  # 请补全

        heiAdd = tf.zeros([inputShape[0], offH, inputShape[2], inputShape[3]])  # 创建 tf.zeros 零矩阵，并赋值给 heiAdd
        weiAdd = tf.zeros([inputShape[0], NewHei, offW, inputShape[3]])  # 请补全
        NewIn = tf.concat([heiAdd, _input, heiAdd], axis=1)  # 创建 tf.concat， 连接 heiAdd,_input,heiAdd， 并赋值给 NewIn
        NewIn1 = tf.concat([weiAdd, NewIn, weiAdd], axis=2)  # 请补全  ***********

        inputNewHei = NewHei  # 将 NewHei 的值赋值给 inputNewHei
        inputNewWid = NewWid  # 将 NewWid 的值赋值给 inputNewWid
        inputNew = NewIn1  # 将 NewIn1 的值赋值给 inputNew
    else:  # 否则
        inputNewHei = inputShape[1]  # 请补全
        inputNewWid = inputShape[2]  # 请补全
        inputNew = _input  # 请补全

    out_height = int(((inputNewHei - filterShape[0]) / _strides[
        1]) + 1)  # 强制转换为整型，int(((inputNewHei - filterShape[0]) / _strides[1]) + 1) ，并赋值给 out_height
    out_width = int(((inputNewWid - filterShape[1]) / _strides[2]) + 1)  # 请补全

    temp = []  # 定义一个空的列表，并赋值给  temp
    for i in range(out_height):  # for 循环遍历， i 从 0，1，..., out_height -1
        for j in range(out_width):  # for 循环遍历， j 从 0，1，..., out_width -1
            temp.append(inputNew[:, i * _strides[1]:i * _strides[1] + filterShape[0],
                        j * _strides[2]:j * _strides[2] + filterShape[1], :])  # 请补全
    print('temp', temp[1])
    temp1 = tf.stack(temp, axis=1)  # 定义 tf.stack ， 并赋值给  temp1
    print('temp1', temp1.shape)
    tempIn = tf.reshape(temp1, [-1, filterShape[0] * filterShape[1] * filterShape[
        2]])  # [batch, out_height, out_width, filterH*filterW*in_channels]
    print('tempIn', tempIn.shape)
    tempFilter = tf.reshape(_filter, [-1, filterShape[3]])  # [filterH*filterW*in_channels,out_channels]
    print('tempFilter', tempFilter.shape)
    dst = tf.matmul(tempIn, tempFilter)  # 定义 tf.matmul，并赋值给 dst
    print('dst', dst.shape)
    dst = tf.reshape(dst, [-1, out_height, out_width, filterShape[3]])  # [batch, out_height, out_width, out_channels]
    return dst  # 返回 dst


def my_max_pool(_input, ksize, strides, padding='VALID'):  # 定义一个 my_max_pool 函数， 输入参数是 _input, ksize, strides, padding
    '''
    #input：输入，4维向量，[batch, in_height, in_width, in_channels]
    #ksize：pool核尺寸,对输入的各个维度，4维向量，[kernelS_batch, kernelS_height, kernelS_width, kernelS_in_channels]
            pool核尺寸可以为奇数或偶数
    #strides：步长，4维向量，针对输入数据的各个维度，stride分别为多少，通常为[1, stride, stride, 1]
    #Padding：边缘扩充，可设置成“SAME”或者“VALID”
    '''
    inputShape = _input.get_shape().as_list()  # get_shape获取一个TensorShape对象，然后通过as_list方法返回每一个维度数

    if padding == "SAME":  # 如果 padding 判断等于 SAME
        NewHei = inputShape[1] + (ksize[1] - 1)  # inputShape 中第二个元素 加上 (ksize[1]-1) ， 赋值给 NewHei
        NewWid = inputShape[2] + (ksize[2] - 1)  # inputShape 中第三个元素 加上 (ksize[2]-1) ， 赋值给 NewWid
        offH = int((ksize[1] - 1) / 2)  # 强制转换为整型，int((ksize[1]-1)/2)，并赋值给 offH
        offW = int((ksize[2] - 1) / 2)  # 强制转换为整型，int((ksize[2]-1)/2)，并赋值给 offW

        heiAdd = tf.zeros([inputShape[0], offH, inputShape[2], inputShape[3]])  # 创建 tf.zeros 零矩阵，并赋值给 heiAdd
        weiAdd = tf.zeros([inputShape[0], NewHei, offW, inputShape[3]])  # 创建 tf.zeros 零矩阵，并赋值给 weiAdd
        NewIn = tf.concat([heiAdd, _input, heiAdd], axis=1)  # 创建 tf.concat， 连接 heiAdd,_input,heiAdd, 并赋值给 NewIn
        NewIn1 = tf.concat([weiAdd, NewIn, weiAdd], axis=2)  # 创建 tf.concat， 连接 weiAdd, NewIn, weiAdd, 并赋值给 NewIn1

        inputNewHei = NewHei  # 将 NewHei 的值赋值给 inputNewHei
        inputNewWid = NewWid  # 将 NewWid 的值赋值给 inputNewWid
        inputNew = NewIn1  # 将 NewIn1 的值赋值给 inputNew
    else:  # 否则
        inputNewHei = inputShape[1]  # 将 inputShape 第二个元素 的值赋值给 inputNewHei
        inputNewWid = inputShape[2]  # 将 inputShape 第三个元素 的值赋值给 inputNewWid
        inputNew = _input  # 将 _input 的值赋值给 inputNew

    out_height = int(((inputNewHei - ksize[1]) / strides[
        1]) + 1)  # 强制转换为整型，int(((inputNewHei - ksize[1]) / strides[1]) + 1)， 赋值给 out_height
    out_width = int(((inputNewWid - ksize[2]) / strides[
        2]) + 1)  # 强制转换为整型，int(((inputNewWid - ksize[2]) / strides[2]) + 1)， 赋值给 out_width

    temp = []  # 定义一个空的列表，并赋值给  temp
    for i in range(out_height):  # for 循环遍历， i 从 0，1，..., out_height -1
        for j in range(out_width):  # for 循环遍历， j 从 0，1，..., out_width -1
            tt = inputNew[:, i * strides[1]:i * strides[1] + ksize[1], j * strides[2]:j * strides[2] + ksize[2],
                 :]  # 赋值给 tt
            # print('1',tt.shape)
            tt = tf.reduce_max(tf.reshape(tt, [-1, strides[0] * strides[1] * strides[2]]), axis=1)  # 请补全
            # print('2',tt.shape)
            temp.append(tt)  # 将 tt 添加到 temp 中
    print('p_temp', temp[1])
    temp1 = tf.stack(temp, axis=1)  # 定义 tf.stack ， 并赋值给 temp1
    print('p_temp1', temp1.shape)
    dst = tf.reshape(temp1, [-1, out_height, out_width, inputShape[3]])  # [batch, out_height, out_width, out_channels]
    return dst  # 返回 dst


def model_cnn(batch_size):  # 定义一个 model_cnn 函数， 构建卷积神经网络模型
    _IMAGE_SIZE = 32  # 图片大小为 32
    _IMAGE_CHANNELS = 3  # 图片通道为 3，对应 RGB
    _conv1Num = 8  # 卷积层一的节点数为 8
    _conv2Num = 24  # 卷积层二的节点数为 24
    _RESHAPE_SIZE = 7 * 7 * _conv2Num  # 修改后的大小为 7*7*_conv2Num
    _fc1Num = 128  # 全链接一层节点数数 128
    _fc2Num = 128  # 全链接二层节点数数 128
    _NUM_CLASSES = 10  # 分类的数量为 10 类
    _BATCH_SIZE = batch_size  # _BATCH_SIZE 的值为 batch_size

    with  tf.name_scope('data'):  # 定义 tf.name_scope
        x = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS],
                           name='images')  # 定义 tf.placeholder ， 并赋值给 x
        y = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _NUM_CLASSES],
                           name='Output')  # 定义 tf.placeholder ， 并赋值给 y

    # conv1

    # 如果name重名，会自动检测命名冲突进行处理
    with tf.name_scope('conv1') as scope:  # 这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
        kernel = get_weight([5, 5, 3, _conv1Num])  # 通过 get_weight 函数得到权重
        conv = my_conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')  # 通过 my_conv2d 函数，构建卷积层
        biases = tf.Variable(tf.constant(0.0, shape=[_conv1Num], dtype=tf.float32), trainable=True,
                             name='bias')  # 定义 tf.Variable 变量， 赋值给偏置 biases
        bias = tf.nn.bias_add(conv, biases)  # 通过 tf.nn.bias_add ， 增加偏置
        conv1 = tf.nn.relu(bias, name=scope)  # 通过 tf.nn.relu， 选择 激活函数 relu, 得到 卷积层一 conv1

        print_activations(conv1)  # 调用 print_activations 函数， 打印出激活值的信息
    tf.summary.histogram('Convolution_layers/conv1', conv1)  # 总结，绘制直方图 tf.summary.histogram
    tf.summary.scalar('Convolution_layers/conver1', tf.nn.zero_fraction(conv1))  # tf.summary.scalar 归一化

    bn1 = tf.layers.batch_normalization(conv1, training=True, name='bn1')  # 调用 tf.layers.batch_normalization 标准化
    # pool1
    pool1 = my_max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 调用 my_max_pool 函数， 最大池化

    print_activations(pool1)  # 调用 print_activations 函数， 打印出激活值的信息

    # conv2
    with tf.name_scope('conv2') as scope:  # 这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
        kernel = get_weight([3, 3, _conv1Num, _conv2Num])  # 通过 get_weight 函数得到权重
        conv = my_conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')  # 通过 my_conv2d 函数，构建卷积层
        biases = tf.Variable(tf.constant(0.0, shape=[_conv2Num], dtype=tf.float32), trainable=True,
                             name='biases')  # 定义 tf.Variable 变量， 赋值给偏置 biases
        bias = tf.nn.bias_add(conv, biases)  # 通过 tf.nn.bias_add ， 增加偏置
        conv2 = tf.nn.relu(bias, name=scope)  # 通过 tf.nn.relu， 选择 激活函数 relu, 得到 卷积层二  conv2

    tf.summary.histogram('Convolution_layers/conv2', conv2)  # 总结，绘制直方图 tf.summary.histogram
    tf.summary.scalar('Convolution_layers/conver2', tf.nn.zero_fraction(conv2))  # tf.summary.scalar 归一化
    print_activations(conv2)  # 调用 print_activations 函数， 打印出激活值的信息

    bn2 = tf.layers.batch_normalization(conv2, training=True, name='bn2')  # 调用 tf.layers.batch_normalization 标准化
    # pool2
    pool2 = my_max_pool(bn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # 调用 my_max_pool 函数， 最大池化
    print_activations(pool2)  # 调用 print_activations 函数， 打印出激活值的信息

    # fully_connected1
    with tf.name_scope('fully_connected1') as scope:  # 这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
        reshape = tf.reshape(pool2, [-1, _RESHAPE_SIZE])  # 调用 tf.reshape 改变形状大小
        dim = reshape.get_shape()[1].value  # 调用 reshape.get_shape 得到维度
        weights = get_weight([dim, _fc1Num])  # 调用 get_weight 得到权重， 并赋值给 weights

        print_activations(weights)  # 调用 print_activations 函数， 打印出激活值的信息
        biases = tf.Variable(tf.constant(0.0, shape=[_fc1Num], dtype=tf.float32), trainable=True,
                             name='biases')  # 定义 tf.Variable 变量， 赋值给偏置 biases
        f1 = tf.matmul(reshape, weights) + biases  # 调用  tf.matmul 相乘
        local3 = tf.nn.relu(f1, name=scope)  # 通过 tf.nn.relu， 选择 激活函数 relu
        print_activations(local3)  # 调用 print_activations 函数， 打印出激活值的信息

    tf.summary.histogram('Fully connected layers/fc1', local3)  # 总结，绘制直方图 tf.summary.histogram
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))  # tf.summary.scalar 归一化

    bn3 = tf.layers.batch_normalization(local3, training=True, name='bn3')  # 调用 tf.layers.batch_normalization 标准化

    # fully_connected2
    with tf.name_scope('fully_connected') as scope:  # 这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
        weights = get_weight([_fc1Num, _fc2Num])  # 调用 get_weight 得到权重， 并赋值给 weights
        print_activations(weights)  # 调用 print_activations 函数， 打印出激活值的信息
        biases = tf.Variable(tf.constant(0.0, shape=[_fc2Num], dtype=tf.float32), trainable=True,
                             name='biases')  # 定义 tf.Variable 变量， 赋值给偏置 biases
        f2 = tf.matmul(bn3, weights) + biases  # 调用  tf.matmul 相乘
        local4 = tf.nn.relu(f2, name=scope)  # 通过 tf.nn.relu， 选择 激活函数 relu
        print_activations(local4)  # 调用 print_activations 函数， 打印出激活值的信息

    tf.summary.histogram('Fully connected layers/fc2', local4)  # 总结，绘制直方图 tf.summary.histogram
    tf.summary.scalar('Fully connected layers/fc4', tf.nn.zero_fraction(local4))  # tf.summary.scalar 归一化

    bn4 = tf.layers.batch_normalization(local4, training=True, name='bn4')  # 调用 tf.layers.batch_normalization 标准化
    # output
    with tf.name_scope('output') as scope:  # 这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
        weights = get_weight([_fc2Num, _NUM_CLASSES])  # 调用 get_weight 得到权重， 并赋值给 weights

        print_activations(weights)  # 调用 print_activations 函数， 打印出激活值的信息
        biases = tf.Variable(tf.constant(0.0, shape=[_NUM_CLASSES], dtype=tf.float32), trainable=True,
                             name='biases')  # 定义 tf.Variable 变量， 赋值给偏置 biases
        softmax_linear = tf.add(tf.matmul(bn4, weights), biases, name=scope)  # 调用 tf.add，tf.matmul， 相乘，相加操作

    tf.summary.histogram('Fully connected layers/output', softmax_linear)  # 总结，绘制直方图 tf.summary.histogram

    global_step = tf.Variable(initial_value=0, name='global_step',
                              trainable=False)  # 定义 tf.Variable 变量， 赋值给 global_step
    y_pred_cls = tf.argmax(softmax_linear, axis=1)  # 调用 tf.argmax ， 取出最大值的下标， 即 预测概率最大的结果

    return x, y, softmax_linear, global_step, y_pred_cls  # 返回 x,y,softmax_linear,global_step,y_pred_cls

