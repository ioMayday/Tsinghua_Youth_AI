import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import utils.config as cfg
from utils.model_yolo import YOLONet
from utils.timer import Timer

#检测的类Detector
class Detector(object):

    def __init__(self, net, weight_file):   # Yolo网络
        '''
        初始化参数和配置模型
        '''
        self.net = net                     # 网络
        self.weights_file = weight_file    # 模型参数文件

        self.classes = cfg.CLASSES         # PASCAL VOC数据集的20个类别
        self.num_class = len(self.classes) # 20
        self.image_size = cfg.IMAGE_SIZE   # 448
        self.cell_size = cfg.CELL_SIZE     # 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL  # 每一个cell预测的框 2
        self.threshold = cfg.THRESHOLD     # 0.2
        self.iou_threshold = cfg.IOU_THRESHOLD  # iou阈值 0.5
        self.idx1 = self.cell_size * self.cell_size * self.num_class  # 7*7*20
        self.idx2 = self.idx1 + self.cell_size * self.cell_size * self.boxes_per_cell    # 7*7*20 + 7*7*2

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())   # 初始化tensorflow中全局变量

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)  # 加载模型参数

    def draw_result(self, img, result):  
        '''
        根据result，向image上画框，result维度为[number, 6], 每组4个数的含义是(class, x_center, y_center, width, height, confidence)
        '''
        print("hell")
        print(len(result))
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img): 
        '''
        处理图像，图像预处理、输入模型检测、
        '''
        ################### 图像预处理 #####################
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))   
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0        
        inputs = np.reshape(inputs, (-1, self.image_size, self.image_size, 3))  #reshape，由于模型输入格式为：[batch_size, image_size, image_size, 3]
        
        ################### 图像检测 #######################
        net_output = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs})  #网络的输出
        print("net_output:",net_output.shape)  #其维度为[batch_size, 7*7*30]
        results = []
        for i in range(net_output.shape[0]): 
            results.append(self.interpret_output(net_output[i]))   

        result = results[0]    # 由于batch_size=1, 所以取结果中的第一个元素
        print(len(result))

        print("输出result1:", result[0][1])
        print("输出result2:", result[0][2])
        print("输出result3:", result[0][3])
        print("输出result4:", result[0][4])

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)  #x_center
            result[i][2] *= (1.0 * img_h / self.image_size)  #y_center
            result[i][3] *= (1.0 * img_w / self.image_size)  #width
            result[i][4] *= (1.0 * img_h / self.image_size)  #height

        return result  #返回框位置，已经是真实坐标了

    def interpret_output(self, output):  
        '''
        进行阈值筛选（筛选的是类别置信度）和进行非极大值抑制
        '''
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))
        
        #读取output中的预测结果
        class_probs = np.reshape( output[0:self.idx1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.idx1:self.idx2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape( output[self.idx2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        
        #因为网络预测出来的是偏移量，因此要恢复
        offset = self.net.offset

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, :2] / self.cell_size  # 得到(x_center, y_cwenter)相对于每一张图片的位置比例
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])              # 得到预测的宽度和高度乘以平方才能得到相对于整张图片的比例

        boxes *= self.image_size      # 得到相对于原图的坐标框

        #计算类别置信度， probs维度为[7, 7, 2, 20]
        for i in range(self.boxes_per_cell):   
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')    # 如果大于self.threshold，那么其对应的位置为true, 否则为false
        filter_mat_boxes = np.nonzero(filter_mat_probs)                       # 找到为true的地方，false是0
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]      # 找到框的位置
        probs_filtered = probs[filter_mat_probs]                              # 找到符合的类别置信度
        
        #若该cell类别置信度大于阈值，则只取类别置信度最大的那个框，一个cell只负责预测一个类别
        classes_num_filtered = np.argmax( filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]  #类别置信度排序，降序排列
        boxes_filtered = boxes_filtered[argsort]  #找到符合条件的框，从大到小排序
        probs_filtered = probs_filtered[argsort]  #找到符合条件的类别置信度，从大到小排序
        classes_num_filtered = classes_num_filtered[argsort]  #类别数过滤

        #非极大值抑制算法， iou_threshold=0.5
        for i in range(len(boxes_filtered)):  
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]  #经过阈值和非极大值抑制之后得到的框
        probs_filtered = probs_filtered[filter_iou]  #经过阈值和非极大值抑制之后得到的类别置信度
        classes_num_filtered = classes_num_filtered[filter_iou]  #经过非极大值抑制之后得到的类别，一个cell只负责预测一个类别

        result = []  #保存的为(classname, x_center, y_center, width, height, confidence)
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2): 
        '''
        计算交叠率，该方法实际上与yoloNet中的一致，也可省去
        '''
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def image_detector(self, imname, wait=0):
        '''
        图像检测，是该类的外部接口
        '''
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()   #记录检测开始的时间
        result = self.detect(image)   #检测
        detect_timer.toc()   #结束检测开始的时间
        
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imwrite('image1.jpg',image)
        #cv2.imshow('Image', image)
        #cv2.waitKey(wait)

#Detector的main函数
def main():
    parser = argparse.ArgumentParser()   #参数解析
    parser.add_argument('--weights', default="yolo.ckpt-6500", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="dataSet", type=str)
    parser.add_argument('--gpu', default='2', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)   #定义网络的框架
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)  #模型文件路径
    detector = Detector(yolo, weight_file)   #初始化Detector类

    # detect from image file
    imname = 'testImg/cat.jpg'  #测试文件
    detector.image_detector(imname)

#main函数
if __name__ == '__main__':
    main()  #调用main函数
