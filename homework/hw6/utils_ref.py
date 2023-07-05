# -*- coding: utf-8 -*-
import tensorflow as tf
slim = tf.contrib.slim

# 定义默认的arg scope
def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    # 指定正则化函数的参数
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # 为卷积层和全连接层的权重设置 weight_decay 
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params) as sc:
            return sc
