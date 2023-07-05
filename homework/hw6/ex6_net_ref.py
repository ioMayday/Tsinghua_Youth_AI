import tensorflow as tf
import utils

slim = tf.contrib.slim

# inception模块
def block_inception(inputs, scope=None, reuse=None):
    # 设定 conv, avg_pool2d,max_pool2d的默认stride=1, padding='SAME'
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], 
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInception', [inputs], reuse=reuse):
            # 定义分支0
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
             # 定义分支1
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 48, [1, 1], scope='Conv2d_0a_1x1')
                '''提示信息：卷积层，输入为上一层的输出，filter个数为96，卷积核尺寸为3×3'''
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')

            # 定义分支2
            with tf.variable_scope('Branch_2'):
                '''提示信息：卷积层，输入为模块的输入，filter个数为64，卷积核尺寸为1×1'''
                branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            # 定义分支3
            with tf.variable_scope('Branch_3'):
                '''提示信息：平均池化层，使用slim，输入为模块的输入'''
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
            # 将分支0-3的输出合并，并作为函数的返回值输出
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

# reduction模块，图像尺寸减半
def block_reduction(inputs, scope=None, reuse=None):

    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], 
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReduction', [inputs], reuse=reuse):
            # 定义分支0
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            # 定义分支1
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            # 定义分支2
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
            # 将分支0-2的输出合并，并作为函数的返回值输出
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

# 定义模型的基本网络结构
def net_base(inputs, scope=None):
    with tf.variable_scope(scope, 'EX6_NET', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')

            # 149 x 149 x 32
            net = slim.conv2d(net, 32, [3, 3], padding='VALID', scope='Conv2d_2a_3x3')

            # 147 x 147 x 32
            net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')

            # 147 x 147 x 64
            with tf.variable_scope('Concat_3a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_0a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])

            # 73 x 73 x 160
            with tf.variable_scope('Concat_4a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID', scope='Conv2d_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])

            # 71 x 71 x 192
            with tf.variable_scope('Concat_5a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1])


            # 35 x 35 x 384
            # 在上述网络结构的基础上，依次添加两个inception模块
            for idx in range(2):
                block_scope = 'Concat_5' + chr(ord('b') + idx)
                net = block_inception(net, block_scope)


            # 35 x 35 x 384
            # Reduction block
            net = block_reduction(net, 'Concat_6a')
            
            # 17 x 17 x 384
            net = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_3a_3x3')
            
            # 8 x 8 x 300
            net = slim.conv2d(net, 300, [3, 3], stride=2, padding='VALID', scope='Conv2d_4a_3x3')
            return net

# 构建完整的网络结构
def ex6_net(inputs, num_classes=20, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='EX6_NET'):
    with tf.variable_scope(scope, 'EX6_NET', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], 
                            is_training=is_training):
            net = net_base(inputs, scope=scope)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], 
                                stride=1, padding='SAME'):
                with tf.variable_scope('Logits'):
                    # 8 x 8 x 300
                    kernel_size = net.get_shape()[1:3]
                    if kernel_size.is_fully_defined():
                            net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                    else:
                        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

                    if not num_classes:
                        return net
                    # 1 x 1 x 300
                    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                    net = slim.flatten(net, scope='PreLogitsFlatten')

                    # 300
                    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')

        return logits

# 默认的图像尺寸
default_image_size = 299
# 默认的arg scope
net_arg_scope = utils.inception_arg_scope
