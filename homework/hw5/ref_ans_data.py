# 模块导入
import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
data_dir = "./dataSet/cifar_10"#所需数据的路径
def _print_download_progress(count, block_size, total_size):#输出下载过程
    pct_complete = float(count * block_size) / total_size#任务完成率
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)#msg的格式为“- Download progress:pct_complete%”
    #\r的作用将转义字符转化为普通字符
    sys.stdout.write(msg)#不换行输出msg
    sys.stdout.flush()
def maybe_download_and_extract():#下载并提取数据
    main_directory = "./dataSet/"#主路径
    cifar_10_directory = main_directory+"cifar_10/"#cifar_10的路径
    if not os.path.exists(main_directory):#判断所给路径是否存在
        os.makedirs(main_directory)#路径不存在则创建新的目录
def maybe_download_and_extract():#下载并提取数据
    main_directory = "./dataSet/"#主路径
    cifar_10_directory = main_directory+"cifar_10/"#cifar_10的路径
    if not os.path.exists(main_directory):#判断所给路径是否存在
        os.makedirs(main_directory)#路径不存在则创建新的目录

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"#下载数据的网址
        filename = url.split('/')[-1]#文件名为cifar-10-python.tar.gz
        file_path = os.path.join(main_directory, filename)#将主路径和文件名合成为一个路径
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
        #urlretrieve()函数将远程数据下载到本地，url为网络地址，filename为本地路径，reporthook是回调函数，显示当前下载进度

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):  # 如果压缩文件后缀为.zip
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)  # 将当前路径下的压缩文件解压
        elif file_path.endswith((".tar.gz", ".tgz")):  # 如果压缩文件的后缀为.tar.gz或.tgz
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)  # 将当前路径下的压缩文件解压
        print("Done.")

        os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)  # 将第一个目录名修改为第二个目录名
        os.remove(zip_cifar_10)  # 删除路径为zip_cifar_10的文件
def one_hot(x, n):#将索引转化为one-hot编码形式
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)#将x转化为数组类型
    assert x.ndim == 1#assert为断言函数，如果布尔表达式的值为False,就会抛出错误
    return np.eye(n)[x]#将x转化为one-hot编码形式，n为每个one-hot数组的组成个数
def get_data_set(name="train", cifar=10):#获取数据集，name的默认值为train，cifar的默认值为10
    x = []
    y = []
    maybe_download_and_extract() #若路径下没有数据集，则下载

    if name is "train":#如果输入参数为train
        for i in range(5):
            f = open(data_dir+'/data_batch_' + str(i + 1), 'rb')#以二进制只读的方式打开文件
            datadict = pickle.load(f, encoding='latin1')  #提取数据
            f.close()#关闭文件

            _X = datadict["data"]#将文件中的样例数据赋值给_X
            _Y = datadict['labels']#将文件中的标签数据赋值给_Y
            _X = np.array(_X, dtype=float) / 255.0#将样例数据转化为float类型的数组，并除以255

            _X = _X.reshape([-1, 3, 32, 32])#将_X转化为3*32*32类型的数组，-1代表reshape函数会根据原有数组维度自动计算出新的数组
            _X = _X.transpose([0, 2, 3, 1])  #由于cifar-10中数据存储格式为R（1024=32*32）/G（1024）/B（1024）
            #将原始数据转化为batch_size*32*32*3

            x.extend(_X)    #list拼接，在x中追加_X中的值
            y.extend(_Y)    #list拼接，在y中追加_y中的值

        x = np.array(x)  # 将x转化为数组类型
        y = np.array(y)  # 将y转化为数组类型

    elif name is "test":#如果输入参数为test
        f = open(data_dir + '/test_batch', 'rb')#以二进制只读的方式打开文件
        datadict = pickle.load(f, encoding='latin1')#提取数据
        f.close()#关闭文件

        x = datadict["data"]#将文件中的样例数据赋值给x
        y = np.array(datadict['labels'])#将文件中的样例数据赋值给

        x = np.array(x, dtype=float) / 255.0#将样例数据转化为float类型的数组，并除以255
        x = x.reshape([-1, 3, 32, 32])#将_X转化为3*32*32类型的数组，-1代表reshape函数会根据原有数组维度自动计算出新的数组
        x = x.transpose([0, 2, 3, 1])   #请补全(将原始数据转化为batch_size*32*32*3)

    return x, one_hot(y, 10)  # 返回x和具有10个数的one-hot数组


