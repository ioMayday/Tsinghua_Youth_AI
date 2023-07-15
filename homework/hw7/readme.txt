
1. 当前目录dataSet文件夹中包含：
   
    数据集：为PASCAL_VOC_2007， dataSet\pascal_voc\VOCdevkit
    训练模型：为train时生成的模型， dataSet\pascal_voc\output
    数据缓存：为训练时，为避免每次都读取训练数据信息，因此生成cache
    权重文件：为需要加载的模型，如训练时预训练模型和测试时模型， dataSet\weights
    
2. 当前目录testImg文件夹中，为测试的图像，供检测当前模型检测的可视化效果展示

3. 当前目录utils文件夹中，为辅助代码

    config.py：配置模型相关参数和文件路径
    data_pascal_voc.py：读取PASCAL_VOC数据
    model_yolo.py：构建模型
    timer.py：时间计算
    
    download_data.sh：若数据集文件加载较慢，可运行该文件，从网上下载数据集
    
4. 当前目录train.py文件，为训练源码文件，该源码调用了utils中的相关文件

5. 当前目录test.py文件，为测试源码文件，该文件调用了utils中的相关文件

注意：为减小CPU的训练慢导致的时间压力，当前程序需要加载预训练模型，该模型已事前在PASCAL_VOC上训练一定次数；
      
      另外，数据集为600张图像，因此建议训练迭代次数小于100次即可