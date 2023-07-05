# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import EX6_NET 
import read_data
slim = tf.contrib.slim
image_size = 299

#获取与synset对应的human readable
synset_to_human_txt="./dataset/train20_metadata.txt"
synset_to_human={}
with open(synset_to_human_txt,'r') as f:
    for line in f:
        part=line.strip('\n').split('\t')
        synset=part[0]
        human=part[1]
        synset_to_human[synset]=human

img_dir="./testimg/n02101388/ILSVRC2012_val_00036408.JPEG"  #待测图像路径

# 获取图像的真实类别名称
synset=img_dir.split("/")[2]
gt=synset_to_human[synset]

with tf.Graph().as_default():
    # 读取图片
    image_string=tf.gfile.FastGFile(img_dir,'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    # 预处理图片
    processed_image = read_data.preprocess_image(image,
                                          image_size,
                                          image_size,
                                          is_training=False)

    processed_images  = tf.expand_dims(processed_image, 0)
    with slim.arg_scope(EX6_NET.net_arg_scope()):
        # 将图片送入模型得到返回结果logits
        logits= EX6_NET.ex6_net(processed_images,
                               num_classes=20,
                               is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        #获取模型
        init_fn = slim.assign_from_checkpoint_fn(( './model/imagenet20'),slim.get_model_variables('EX6_NET'))
        with tf.Session() as sess:

            # 加载模型
            init_fn(sess)
            # 图片预处理，以numpy矩阵的格式传入网络模型，执行前向计算
            np_image, network_input, probabilities = sess.run([image,
                                                               processed_image,
                                                               probabilities])

            # 获取类别对应的名称，用于打印测试结果
            names = read_data.create_readable_names_for_imagenet_labels()
            
            #打印top5的标签，名称和概率值
            predictions = np.squeeze(probabilities)
            print("size= {}".format(len(predictions)))
            print("real label is {}".format(gt))
            top_k = predictions.argsort()[-5:][::-1] # 获取top5的索引

            top_names = []
            for node_id in top_k:
                human_string = names[node_id] 
                top_names.append(human_string)
                score = predictions[node_id]
                print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
           
 
            
