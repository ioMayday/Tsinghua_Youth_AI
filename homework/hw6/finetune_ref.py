# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import read_data
import EX6_NET
import tensorflow.contrib.slim as slim

LEARNING_RATE = 0.0001   # 设置初始学习率
N_CLASSES = 20   # 20类（来自imagenet的1000类）
IMAGE_WEITH = 299 # 输入图像宽度
IMAGE_HEIGHT = 299 # 输入图像高度
BATCH_SIZE = 32   # 设置batch大小
EPOCH = 2  # 设置训练次数
num_threads=4  # 使用多线程读取batch

dataset_dir = "./dataset/ILSVRC2012_20_tfrecord"  # tfrecord格式数据集所在路径
model_path="./savedmodels/imagenet20"  # 保存模型的路径（前缀）
finetune_model="./model/imagenet20"  # 加载预训练模型的路径

tf.reset_default_graph() # 开始运行时，重启tensorflow图，解决多次运行报错。



# 计算精度
def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct) * 100.0
    return acc

def main():
    # 获取训练样本集
    train_image_bath, train_label_batch=read_data.get_train_batch_images_and_label(dataset_dir,BATCH_SIZE,N_CLASSES,True,IMAGE_HEIGHT,IMAGE_WEITH,num_threads)
    # 获取验证样本集
    val_image_bath, val_label_batch=read_data.get_val_batch_images_and_label(dataset_dir,BATCH_SIZE,N_CLASSES,False,IMAGE_HEIGHT,IMAGE_WEITH,num_threads)
    
    train_img_num=read_data._SPLITS_TO_SIZES['train'] # 训练样本总数
    print("train_img_num = {}".format(train_img_num))
    batch_total = int(train_img_num / BATCH_SIZE) + 1  # batch总数

    # 设置输入占位符
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_WEITH, IMAGE_HEIGHT, 3], name="input_images")
    labels = tf.placeholder(tf.int16, shape=[None, N_CLASSES], name="labels")  # one_hot

    global_steps = tf.Variable(0, trainable=False)

    # 前向计算
    with slim.arg_scope(EX6_NET.net_arg_scope()):
        logits = EX6_NET.ex6_net(images, num_classes=N_CLASSES)

    # 获取训练参数名称（可选项）
    variable_name = [v.name for v in tf.trainable_variables()]
    
    # 定义损失和优化方法
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).\
            minimize(total_loss, global_step=global_steps)
    
    # 计算精度
    acc = accuracy(logits, labels)


    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(
        finetune_model,
        slim.get_model_variables('EX6_NET'))

    # 模型保存接口
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 开启一个协调器，用于多线程管理
        threads = tf.train.start_queue_runners(sess, coord)  # 入队线程启动器
        
        # 初始参数
        init = tf.global_variables_initializer() 
        sess.run(init)

        # 加载已经预训练好的模型
        print("Loading tuned variables from %s" % finetune_model)
        load_fn(sess)
        
        #打印变量名称
        values = sess.run(variable_name)
        for k,v in zip(variable_name, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

        try:
            for i in range(EPOCH):  # 每一轮迭代
                if coord.should_stop():
                    break
                print('****** Epoch{}/{} ******'.format(i, EPOCH))
                
                # 依次训练每个batch
                for step in range(batch_total): 
                    # 获取每一个batch中batch_size个样本和标签
                    images_batch, labels_batch = sess.run([train_image_bath, train_label_batch])
                    _, tra_loss, tra_acc = sess.run([train_step, total_loss, acc],
                                                    feed_dict={images: images_batch,
                                                               labels: labels_batch})
                    # 打印训练集的损失和精度
                    if step % 10 == 0 or (step + 1) == batch_total:
                        tra_loss = format(tra_loss, "3.3f")
                        tra_acc = format(tra_acc, "3.3f")
                        print('{}/{}[************************************] - train_loss: {} train_acc: {}%'
                              .format(step, batch_total, tra_loss, tra_acc))
                    
                    # 打印验证集的损失和精度，并保存模型
                    if step % 50 == 0 or (step + 1) == batch_total:
                        val_images, val_labels = sess.run([val_image_bath, val_label_batch])
                        val_loss, val_accuracy = sess.run([total_loss, acc],
                                                          feed_dict={images: val_images,
                                                                     labels: val_labels})
                        val_loss = format(val_loss, "3.3f")
                        val_accuracy = format(val_accuracy, "3.3f")
                        print('{}/{}[====================================] - val_loss: {} val_acc: {}%'
                              .format(step, batch_total, val_loss, val_accuracy))
                        saver.save(sess, model_path, global_step=global_steps, write_meta_graph=True)

        except tf.errors.OutOfRangeError:   # 如果读取到文件队列末尾会抛出此异常
            print("done")
        finally:
            coord.request_stop()   # 协调器coord发出所有线程终止信号
        coord.join(threads) # 把开启的线程加入主线程，等待threads结束

if __name__ == "__main__":
    main()


