import numpy as np
import matplotlib.pyplot as plt
import xlrd     # 导入必备的 xlrd 库，目的是为了调用 xlrd.open_workbook 函数打开 excel 文件，读取数据

class Config:
    input_dim = 2  # input layer dimensionality
    output_dim = 1  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    lr = 5  # learning rate for gradient descent
    reg_lambda = 0.01 #0.01  # regularization strength

# 定义函数loadData函数，输入参数是 filename 指代文件名，返回数据data，目的是从.xls文件中加载数据，并存储为numpy中的array格式
def loadData(filename):
    workbook = xlrd.open_workbook(filename)         # 通过调用 xlrd.open_workbook 函数打开 excel 文件，读取数据，并返回给 workbook 变量
    boyinfo = workbook.sheet_by_index(0)            # 通过使用属性 sheet_by_index 得到  excel 文件 中的工作簿，其中 sheet_by_index(0) 表示是第一个工作簿，在 python 中，下标从 0 开始
    col_num = boyinfo.ncols                         # 通过使用属性 ncols 得到 excel 文件 中第一个工作簿的 列数，并赋值给 col_num
    row_num = boyinfo.nrows                         # 通过使用属性 nrows 得到 excel 文件 中第一个工作簿的 行数，并赋值给 row_num
    col0 = boyinfo.col_values(0)[1:]                # 通过使用属性 col_values(0)[1:] 得到 excel 文件 中第一列数据中，从第2行到最后一行的所有数据，并赋值给 col0
    data = np.array(col0)                           # 通过使用 np.array 函数， 将 col0 转换成数组，并赋值给 data
    if col_num == 1:                                # 条件判断语句： 如果列数 col_num 为1， 只有一列，那么直接返回数据 data
        return data                                     # 返回data
    else:                                           # 否则，如果不止一列数据，需要遍历所有列的数据
        for i in range(col_num-1):                      # 通过使用for循环达到遍历的目的
            coltemp = boyinfo.col_values(i+1)[1:]           # 从第二行开始，表头不算，遍历从 第二列 开始到最后一列的数据
            data = np.c_[data, coltemp]                     # 通过使用 np.c_ 函数将 第一列的数据 和后面 所有列的数据组合起来，并赋值给 data
    return data                                     # 返回data

# 定义一个 plotData 函数，输入参数是 数据 X 和标志 flag: y，返回作图操作 plt, p1, p2 ， 目的是为了画图
def plotData(X, y):
    pos = np.where(y==1)                            # 通过使用 np.where 函数查找所有满足条件的数据，查找所有满足标志 y == 1 的数据，并赋值给 pos
    neg = np.where(y==0)                            # 通过使用 np.where 函数查找所有满足条件的数据，查找所有满足标志 y == 0 的数据，并赋值给 neg
    # 通过使用 plt.plot 函数作图，对所有满足标志 y == 1 的数据作图，点采用 s (正方形)，代表 square, 点的大小为 7 单位，颜色为 红色 red
    p1 = plt.plot(X[pos, 0], X[pos, 1], marker='s', markersize=3, color='red')[0]
    # 通过使用 plt.plot 函数作图，对所有满足标志 y == 1 的数据作图，点采用 o (圆形)，代表 circle, 点的大小为 7 单位，颜色为 绿色 green
    p2 = plt.plot(X[neg, 0], X[neg, 1], marker='o', markersize=3, color='green')[0]
    return plt,p1,p2                            # 返回作图操作plt, p1, p2

# normalization： 定义一个 normalization 函数，输入参数是原始数据 X ，返回归一化后的数据 X_norm ， 目的是为了数据预处理，得到归一化后的数据 X_norm
def normalization(X):
    mu = np.mean(X, axis=0)  # 对数据X的每列求均值，axis = 0 代表在矩阵第一个维度上求均值
    Xmin = np.min(X, axis=0)  # 对数据X的每列求最小值，axis = 0 代表在矩阵第一个维度上求最小值
    Xmax = np.max(X, axis=0)  # 对数据X的每列求最大值，axis = 0 代表在矩阵第一个维度上求最大值
    X_norm = (X-mu)/(Xmax-Xmin)  # 计算归一化后的数据，归一化公式为：(2*(X-Xmin)/(Xmax-Xmin))-1，归一化后数据范围为 [-1,1]
    return X_norm  # 返回数据预处理，归一化后的数据 X_norm


# visualize: 定义一个visualize函数，输入参数为特征矩阵X，标签数据y和分类模型model，函数的作用是：展示出模型的的分类边界
def visualize(X, y, model):
    plot_decision_boundary(lambda x:predict(model,x), X, y)  # 调用plot_decision_boundary函数
    plt.savefig("result.png")  # 将plot_decision_boundary函数中返回的图片保存，图片名字为result.png
    plt.show()  # 展示plot_decision_boundary中返回的图片


# plot_decision_boundary:定义决策边界函数，输入为预测函数 特征矩阵数据X 标签数据y
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5  # 将特征矩阵X中第一列中最小值与最大值分别加上0.5 赋予x_min,x_max
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5  # 将特征矩阵X中第二列中最小值与最大值分别加上0.5 赋予y_min,y_max
    h = 0.01  # 步长为0.01
    # Generate a grid of points with distance h between them
    # 由np.arrange生成一维数组作为np.meshgrid的参数，返回xx矩阵，yy矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    # .ravel()方法将xx,yy矩阵压缩为一维向量；np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    # 合成的矩阵作为pred_func的输入，返回预测值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z.T).reshape(xx.shape)  # Z矩阵转置并将维度调整和xx的维度一致
    p=plt.figure()  # 生成一个“画布”
    _,p1,p2=plotData(X,y)  # 将特征矩阵X与标签数据y传入plotData函数，返回图操作p1,p2,其中‘_’用来接没有用到的返回值
    p3=plt.contour(xx, yy, Z, levels=0,linewidths=2).collections[0]  # 画登高线，即决策边界
    # label & Legend, specific for the exercise
    plt.xlabel("tall")  # 横坐标的标签为tall
    plt.ylabel("salary")  # 纵坐标的标签为salary
    plt.legend((p1, p2, p3), ('y = I like you', "y = I don't like you", 'Decision Boundary'), numpoints=1,handlelength=0)  # 为每一个绘图添加图例
    plt.title("ANN")  # 设置图标题


# 定义sigmoid激活函数，将输入数据压缩在0-1之间
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))  # 根据sigmoid 函数公式写出
    return g  # 返回函数输出值


# 定义sigmoidGradient函数，计算sigmoid函数的梯度值
def sigmoidGradient(z):
    g = sigmoid(z)
    g = g * (1 - g)  # 根据sigmoid 函数公式写出
    return g  # 返回梯度值

def add_row(a):
    a = np.vstack((np.ones((1, a.shape[1])), a))
    return a


# Helper function to evaluate the total loss on the dataset
# 定义损失函数，计算所有样本的损失值
def calculate_loss(model, X, y):
    num_examples = X.shape[1]  # training set size  # X的第二个维度为训练集样本个数
    W1, W2, W3,= model['W1'], model['W2'] , model['W3']  # 神经网络为两层隐藏层，对应的参数矩阵分别为W1 W2 W3
    # Forward propagation to calculate our predictions 需要补全
    a1 = X
    z2 = np.dot(W1,a1)  #  参数W1与a1做矩阵乘法
    a2 = sigmoid(z2)  # 对矩阵z2进行做sigmoid激活
    a2 = add_row(a2)  # 为第一层隐藏层的矩阵a2增加一列值为1的偏置
    z3 = np.dot(W2,a2)  # 参数W2与a2做矩阵乘法
    a3 = sigmoid(z3)  # 对矩阵z3做sigmoid激活
    a3 = add_row(a3)  # 为第二层隐藏层的矩阵a3增加一列值为1的偏置
    z4 = np.dot(W3,a3)  # 参数W3与a3做矩阵乘法
    a4 = sigmoid(z4)  # 对矩阵z4做sigmoid激活

    # Calculating the loss
    one = np.multiply(y, np.log(a4))  # 将真实标签y与预测值a4的对数值对应相乘
    two = np.multiply((1 - y), np.log(1-a4))  # 将真实标签（1-y)与预测值（1-a4）对数值对应相乘
    data_loss = -(1. / num_examples) * (one + two).sum()  # 损失函数的和，对应交叉熵公式
    return data_loss  # 返回损失值


# 定义compare函数，将预测值大于0.5的归为正例，小于0.5的归为负例
def compare(X):
    num = X.shape[1]  # X的第二个维度为训练集样本个数，注意X为函数的形参，真正数据调用时传入的实参
    result = []  # 声明一个存放结果的列表
    for i in range(num):  # 遍历所有结果
        if X[:,i]>0.5:  # 判断预测结果是否大于0.5
            result.append(1.0)  # 如果大于0.5，则在result列表中增加一个1.0
        else:
            result.append(0.0)  # 否则在result列表中增加一个0.0
    return result  # 返回result列表，里面是预测为正例与反例的结果


# 定义predict预测函数，输入未训练好的模型和特征矩阵X，返回预测值
def predict(model, X):
    m = X.shape[0]  # 将输入矩阵的第一个维度赋值给m
    W1, W2, W3= model['W1'], model['W2'] , model['W3']  #  将模型训练好的参数分别赋值给W1 W2 W3
    # Forward propagation 需要补全
    X_m = np.transpose(np.column_stack((np.ones((m, 1)), X)))  # 为输入矩阵增加一列值为1的偏置
    a1= X_m  # 将矩阵X_m赋予a1
    # a1 = X
    z2 = np.dot(W1,a1)  #  参数W1与a1做矩阵乘法
    a2 = sigmoid(z2)  # 对矩阵z2进行做sigmoid激活
    a2 = add_row(a2)  # 为第一层隐藏层的矩阵a2增加一列值为1的偏置
    z3 = np.dot(W2,a2)  # 参数W2与a2做矩阵乘法
    a3 = sigmoid(z3)  # 对矩阵z3做sigmoid激活
    a3 = add_row(a3)  # 为第二层隐藏层的矩阵a3增加一列值为1的偏置
    z4 = np.dot(W3,a3)  # 参数W3与a3做矩阵乘法
    a4 = sigmoid(z4)  # 对矩阵z4做sigmoid激活
    return a4  # 返回输出矩阵


# 定义precision函数：输入为训练模型，与特征矩阵，目的是返回样本预测结果，正例为1，反例为0
def precision(model, X):
    W1, W2, W3= model['W1'], model['W2'],  model['W3']  # 将模型更新后的参数赋值给W1 W2 W3
    # Forward propagation 需要补全
    a1 = X
    z2 = np.dot(W1,a1)  #  参数W1与a1做矩阵乘法
    a2 = sigmoid(z2)  # 对矩阵z2进行做sigmoid激活
    a2 = add_row(a2)  # 为第一层隐藏层的矩阵a2增加一列值为1的偏置
    z3 = np.dot(W2,a2)  # 参数W2与a2做矩阵乘法
    a3 = sigmoid(z3)  # 对矩阵z3做sigmoid激活
    a3 = add_row(a3)  # 为第二层隐藏层的矩阵a3增加一列值为1的偏置
    z4 = np.dot(W3,a3)  # 参数W3与a3做矩阵乘法
    a4 = sigmoid(z4)  # 对矩阵z4做sigmoid激活
    result = compare(a4)  # 调用compare函数，返回预测结果
    return result  #  返回预测结果


# 定义randInitializeWeights，参数为输入维度和输出维度，作用是随机初始化参数矩阵
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))  #生成一个维度为(L_out, 1 + L_in)的全0矩阵
    # Randomly initialize the weights to small values
    epsilon_init = 0.12  # 初始化一个很小的数
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init  #  随机生成维度为(L_out, 1 + L_in)的参数矩阵
    return W  # 返回参数矩阵


# only for this model
# X：输入样本
# m：样本个数
# Wx：每层权重参数
def nn_fp_proc(X, m, W1, W2, W3):
    # tbd optmazation
    X_m = np.transpose(np.column_stack((np.ones((m, 1)), X)))  # 为输入矩阵增加一列值为1的偏置
    a1 = X_m  # 将矩阵X_m赋予a1
    z2 = np.dot(W1,a1)  #  参数W1与a1做矩阵乘法
    a2 = sigmoid(z2)  # 对矩阵z2进行做sigmoid激活
    a2 = add_row(a2)  # 为第一层隐藏层的矩阵a2增加一列值为1的偏置
    z3 = np.dot(W2,a2)  # 参数W2与a2做矩阵乘法
    a3 = sigmoid(z3)  # 对矩阵z3做sigmoid激活
    a3 = add_row(a3)  # 为第二层隐藏层的矩阵a3增加一列值为1的偏置
    z4 = np.dot(W3,a3)  # 参数W3与a3做矩阵乘法
    a4 = sigmoid(z4)  # 对矩阵z4做sigmoid激活
    return a4


# This function learns parameters for the neural network and returns the model.
# - hidden1_dim: Number of nodes in the hidden layer 1
# - hidden2_dim: Number of nodes in the hidden layer 2
# - iterNum: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
# 定义build_model函数，输入为特征矩阵X，标签向量y，第一层隐藏层神经元个数，第二层隐藏层神经元个数，迭代次数，是否打印损失函数的布尔变量
# 作用是完成神经网络的前向和反向传播，训练参数W1 W2 W3
def build_model(X, y, hidden1_dim, hidden2_dim, iterNum=2000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    m = X.shape[0]  #将输入矩阵X的第一个维度赋予m
    
    W1 = randInitializeWeights(Config.input_dim, hidden1_dim)  # 调用randInitializeWeights函数，初始化W1
    W2 = randInitializeWeights(hidden1_dim, hidden2_dim)  # 调用randInitializeWeights函数，初始化W2
    W3 = randInitializeWeights(hidden2_dim, Config.output_dim)  # 调用randInitializeWeights函数，初始化W3

    # This is what we return at the end
    model = {}  # 将model声明为字典数据格式
    # Gradient descent.
    logName = "logText.txt"  # 日志文件名称
    logFile = open(logName, "w")  # 调用open函数,打开文件，模式为写
    for t in range(0, iterNum):  # 从0循环至iterNum
        # Forward propagation 需要补全
        # a输出在回溯计算时需要用到，最好设成全局变量，再提炼到函数里
        X_m = np.transpose(np.column_stack((np.ones((m, 1)), X)))  # 为输入矩阵增加一列值为1的偏置
        a1 = X_m  # 将矩阵X_m赋予a1
        z2 = np.dot(W1,a1)  #  参数W1与a1做矩阵乘法
        a2 = sigmoid(z2)  # 对矩阵z2进行做sigmoid激活
        a2 = add_row(a2)  # 为第一层隐藏层的矩阵a2增加一列值为1的偏置
        z3 = np.dot(W2,a2)  # 参数W2与a2做矩阵乘法
        a3 = sigmoid(z3)  # 对矩阵z3做sigmoid激活
        a3 = add_row(a3)  # 为第二层隐藏层的矩阵a3增加一列值为1的偏置
        z4 = np.dot(W3,a3)  # 参数W3与a3做矩阵乘法
        a4 = sigmoid(z4)  # 对矩阵z4做sigmoid激活

        # Back propagation 需要补全
        y_m = np.transpose(np.reshape(y, [-1, 1])) #reshape y_m from (n,)to (1,n)
        delta4 = a4 - y_m  # 计算delta4，将预测标签向量a4与y_m做差
        #print(sigmoidGradient(z3).shape)
        A3 = np.dot(W3.T,delta4) # A3第一行元素为偏置层
        delta3 = np.multiply(np.delete(A3,0,axis=0),sigmoidGradient(z3)) # 计算delta3，参数矩阵W3转置后与delta4做矩阵乘法，然后与sigmoidGradient(z3)对应位相乘
        #print(delta3.shape)
        A2 = np.dot(W2.T,delta3)
        delta2 = np.multiply(np.delete(A2,0,axis=0),sigmoidGradient(z2))  # 计算delta2，参数矩阵W2转置后与delta3做矩阵乘法，然后与sigmoidGradient(z2)对应位相乘

        # 计算Dij/Wij
        delta4 = np.array(delta4)
        a3 = np.array(a3)
        # bigDelta3 = delta4 * a3.T # 点乘
        bigDelta3 = np.dot(delta4, a3.T) # 矩阵乘
        DW3 = 1./m * bigDelta3
        W3 += -Config.lr * DW3
        
        bigDelta2 = np.dot(delta3, a2.T)
        DW2 = 1./m * bigDelta2
        W2 += -Config.lr * DW2
        
        bigDelta1 = np.dot(delta2, a1.T)
        DW1 = 1./m * bigDelta1
        W1 += -Config.lr * DW1

        # Assign new parameters to the model
        model = {'W1': W1, 'W2': W2, 'W3': W3}  #模型的键值对分别对应更新后的参数W1 W2 W3

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and t % 1000 == 0:  #如果print_loss 与 t是1000的整数倍同时为True，运行下面代码
            print("Loss after iteration %i: %f" % (t, calculate_loss(model, X_m, y_m)))   #格式化打印语句，输出迭代t次后，损失值是多少
            logFile.write("Loss after iteration %i: %f" % (t, calculate_loss(model, X_m, y_m)))  # 将输出语句写入日志文件
            logFile.write("\n")
            result = precision(model, X_m)  #调用precision函数，返回预测结果
            print("Traning Set Accuracy: {:f}".format((np.mean(result == y) * 100)))  #计算准确率
            logFile.write("Traning Set Accuracy: {:f}".format((np.mean(result == y) * 100)))  # 将输出语句写入日志文件
            logFile.write("\n")  # 换行
    logFile.close()  # 关闭文件

    return model  # 返回模型，实际是返回模型更新后的参数

def main():
    # load data 加载数据
    data = loadData('data.xls')  # 通过调用 loadData 函数，导入原始数据集 文件 'data.xls'，并赋值给 data
    X = data[:, :2]  # 将数据集 data 的 第一列 和 第二列 的所有行的数据，赋值给　Ｘ, 实际对应的是 身高（m）、　月薪（元）的原始数据
    y = data[:, 2]  # 将数据集 data 的 第三列 所有行的数据，赋值给　y，实际对应的是 是否有兴趣尝试交往（Y=1/N=0）的原始数据，可取 0 或 1
    # normalization 通过调用 normalization 函数，对原始数据集 X 进行归一化
    X_norm = normalization(X)
    #print(X_norm.shape)
    #print(y.shape)

    # 训练模型
    hidden1_dim = 5
    hidden2_dim = 3
    model = build_model(X_norm, y, hidden1_dim, hidden2_dim, iterNum=20000, print_loss=True)

    # 可视化
    visualize(X_norm, y, model)


if __name__ == "__main__":
    main()
