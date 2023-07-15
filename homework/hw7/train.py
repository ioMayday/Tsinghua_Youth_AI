# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import tensorflow as tf
import utils.config as cfg
from utils.model_yolo import YOLONet
from utils.timer import Timer
from utils.data_pascal_voc import pascal_voc

tf.reset_default_graph()
slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):   #Yolon_net and pascal_voc_data
        '''
        初始化参数和配置模型
        '''
        self.net = net        #训练的网络
        self.data = data      #train或者test的数据
        
        ########## config文件信息读取和保存 #############
        self.weights_file = cfg.WEIGHTS_FILE    #权重文件
        self.max_iter = cfg.MAX_ITER            #迭代次数,迭代次数可自定义
        self.initial_learning_rate = cfg.LEARNING_RATE     #学习率，0.0001
        self.decay_steps = cfg.DECAY_STEPS      #学习率衰变步数
        self.decay_rate = cfg.DECAY_RATE        #学习率衰变率
        self.staircase = cfg.STAIRCASE          #学习率是否为离散式变化 true
        self.summary_iter = cfg.SUMMARY_ITER    #信息提交频率
        self.save_iter = cfg.SAVE_ITER          #模型保存频率
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))  # 结果保存的路径，按时间命名， data/pascal_voc/output/date_time
        if not os.path.exists(self.output_dir): #不存在则创建目录
            os.makedirs(self.output_dir)
        self.save_cfg()                         #保存配置

        ################# 模型配置 ##################
        self.variable_to_restore = tf.global_variables()                          #初始化tensorflow的全局变量
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=4)      #定义tf.saver，用于模型保存,最多保存最新的4个
        self.ckpt_file = os.path.join(self.output_dir, 'yolo.ckpt')               #定义保存模型的文件名
        self.summary_op = tf.summary.merge_all()                                  #将tensorboard的summary信息汇总
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)       #将summary信息写入到文件中，每60秒更新一次

        self.global_step = tf.train.create_global_step()                          #创建全局的step
        self.learning_rate = tf.train.exponential_decay(                          #设定变化的学习率 decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) #采用的优化方法是adam
        self.train_op = slim.learning.create_train_op(                            #将tensorflow的operation联合起来
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()                                             
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())                          #初始化全局变量

        #加载权重
        if self.weights_file is not None:  
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self): 
        '''
        训练
        '''
        train_timer = Timer()  #train_timer
        load_timer = Timer()   #load_timer

        # 加载验证数据
        val_images, val_labels = self.data.getValid()
        val_feed_dict = {self.net.images: val_images, self.net.labels: val_labels}  
        
        # 迭代训练
        for step in range(1, self.max_iter + 1):
            #print("step: ",step)
      
            load_timer.tic()
            train_images, train_labels = self.data.getTrain() #获取到batch_size大小的图片和对应的label
            load_timer.toc()
            
            #################   开始训练   ################
            train_feed_dict = {self.net.images: train_images, self.net.labels: train_labels}  
            train_timer.tic()
            summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=train_feed_dict)
            train_timer.toc()

            if step % self.summary_iter == 0:  # 记录summary信息
                self.writer.add_summary(summary_str, step)
                
                if step % (self.summary_iter * 10) == 0:   #记录当前训练的模型信息
                    [loss_train] = self.sess.run([self.net.total_loss],feed_dict=train_feed_dict)
                    [loss_val] = self.sess.run([self.net.total_loss],feed_dict=val_feed_dict)
                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {}, Loss-train: {:5.3f}, 
                        Loss-val: {:5.3f}, Speed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}
                        '''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'), self.data.epoch,
                        int(step), round(self.learning_rate.eval(session=self.sess), 6),
                        loss_train, loss_val,
                        train_timer.average_time, load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

            if step % self.save_iter == 0:   #保留检查点，以供测试时用
                print('{} Saving checkpoint file to: {}'.format(datetime.datetime.now().strftime('%m-%d %H:%M:%S'), self.output_dir))
                self.saver.save( self.sess, self.ckpt_file, global_step=self.global_step)
                print("save done!!!")
                
    def save_cfg(self):
        '''
        把配置信息写入到文件中, 保存
        '''
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:   
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):    #更新配置文件路径
    '''
    更新配置文件中的相关数据路径
    '''
    print("应该是加载了预训练模型")
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():    #自定义参数
    '''
    主函数，可在执行程序时，通过command的形式添加参数，也可修改配置文件
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="", type=str)  #定义权重文件
    parser.add_argument('--data_dir', default="dataSet", type=str)         #定义数据文件夹
    parser.add_argument('--threshold', default=0.2, type=float)            #阈值
    parser.add_argument('--iou_threshold', default=0.5, type=float)        #IOU阈值
    parser.add_argument('--gpu', default='0', type=str)                    #gpu训练配置信息
    args = parser.parse_args()

    if args.gpu is not None: 
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()                 #Yolo网络
    pascal = pascal_voc('train')     #获得训练的数据， 包含了经过水平翻转后的训练实例

    solver = Solver(yolo, pascal)    #准备训练的环境

    print('Start training ...')
    solver.train()                   #start training
    print('done!!!')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()  # main 函数
