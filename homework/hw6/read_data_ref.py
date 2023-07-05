# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
slim = tf.contrib.slim



_FILE_PATTERN = '%s-*'   # 数据集文件格式

# 训练集和验证集的数量
_SPLITS_TO_SIZES = {
    'train': 26000,
    'validation': 1000,
}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
}


_NUM_CLASSES = 20 #无背景类



#用于获取Dataset类
# split_name ："train" or "validation"
# dataset_dir: 数据集路径
# metadata_txt：txt文件路径, 该文件记录synset和human readable label 对应关系
def get_data(split_name, dataset_dir, metadata_txt='./dataset/train20_metadata.txt'):
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
      }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    
    
    # 将0-19的标签与相应类别名称保存到字典
    labels_to_names ={}
    label_index=0
    with open(metadata_txt,'r') as f:
        for line in f:
            row_line=line.strip('\n').split('\t',1)
            name = row_line[1]
            labels_to_names[label_index]=name
            label_index+=1

    return slim.dataset.Dataset(
        data_sources=file_pattern,    # 数据源列表
        reader=tf.TFRecordReader,     # 阅读器, 能读取数据源
        decoder=decoder,              # 解码器, 用来解码阅读器返回的每一个样例
        num_samples=_SPLITS_TO_SIZES[split_name],    #样例总数量
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,   # 可选字典, 该字典映射返回列表和相应描述
        num_classes=_NUM_CLASSES,    #类别数量
        labels_to_names=labels_to_names)  # 标签与名称


# 定义随机选择器
def apply_with_random_selector(x, func, num_cases):
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

#定义颜色变换函数，用于数据增强
def distort_color(image, color_ordering=0, scope=None):
  with tf.name_scope(scope, 'distort_color', [image]):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)

# 训练集预处理
def preprocess_for_train(image, height, width):
    # 将图像数据转换为floats.
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    # resize图像尺寸
    distorted_image = apply_with_random_selector(
        image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=1)
    
    # 随机水平镜像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    # 随机颜色变换
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=1)

    #将图片的像素转换到[-1,1]区间内
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image
    
# 验证集预处理
def preprocess_for_eval(image, height, width,central_fraction=0.875):
    # 将图像数据转换为floats.
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    # 获取指定比例的图像
    if central_fraction:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

    # resize图像尺寸
    if height and width:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
        image = tf.squeeze(image, [0])
    
    #将图片的像素转换到[-1,1]区间内
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

# 图像处理函数
def preprocess_image(image, output_height, output_width, is_training=False):
    if is_training:
        return preprocess_for_train(
        image, output_height, output_width)
    else:
        return preprocess_for_eval(
        image, output_height, output_width)

# 获取训练集的batch
def get_train_batch_images_and_label(dataset_dir,batch_size,num_classes,is_training=True,output_height=224, output_width=224,num_threads=4):
    dataset=get_data('train',dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=2,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    [image, label] = provider.get(['image', 'label'])

    image=preprocess_image(image,output_height,output_width,is_training)
    images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes)
    return images,labels

# 获取验证集的batch
def get_val_batch_images_and_label(dataset_dir,batch_size,num_classes,is_training=False,output_height=224, output_width=224,num_threads=4):
    dataset=get_data('validation',dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)

    [image, label] = provider.get(['image', 'label'])

    image=preprocess_image(image,output_height,output_width,is_training)
    images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes)
    return images,labels

# 获取类别标签和名称的对应关系
# 如 0：jay  
def create_readable_names_for_imagenet_labels():
  filename="./dataset/train20_synset.txt"    # synset文件所在路径
  
  #获取synset列表
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 20

  filename="./dataset/train20_metadata.txt"  # metadata文件所在路径
  
  #获取synset和名称的对应关系
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 20 
  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  # 获取类别标签和名称的对应关系
  label_index = 0
  labels_to_names={}

  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  return labels_to_names