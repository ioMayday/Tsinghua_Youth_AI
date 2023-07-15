import os

#
# path and dataset parameter
#配置文件
#

DATA_PATH = 'YOLO_out/dataset'
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')   #存放输出文件的地方，data/pascal_voc/output
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')  #weights_dir, 路径为data/pascal_voc/weights

#WEIGHTS_FILE = None   #weights file
WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'yolo.ckpt-6500')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']    #PASCAL VOC数据集的20个类别

FLIPPED = True    


#
# model parameter
#

IMAGE_SIZE = 448      #输入图片的尺寸
CELL_SIZE = 7         #整张图片分为cell_size * cell_size的大小
BOXES_PER_CELL = 2    #每个cell负责预测两个bounding box
ALPHA = 0.1           #激活函数负半轴保留比例
DISP_CONSOLE = False

#以下为loss用到的参数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''                 #GPU编号，若为多GPU，可设置该值
LEARNING_RATE = 0.0001   #学习率
DECAY_STEPS = 3000       #learning rate 参数，决定lr的减小频率
DECAY_RATE = 0.1         #learning rate 参数，决定lr的减小幅度
STAIRCASE = False        #若为False, 则learning rate 每个step均减小；否则，每隔DECAY_STEPS减小一次
BATCH_SIZE = 64         #batch size
MAX_ITER = 10        #迭代次数
SUMMARY_ITER = 10        #summary训练信息的频率
SAVE_ITER = 500          #保存模型的频率


#
# test parameter
#

THRESHOLD = 0.2       #（类别*box置信度）的阈值
IOU_THRESHOLD = 0.5   #IOU阈值0.5
