import matplotlib.pyplot as plt
from numpy import *


#加载数据集
def loadDataSet():
    '''数据集的前两个值分别为X1和X2,第三个值是数据对应的类别标签，为了方便计算，把X0的值设置成了1.0'''
    dataMat = []
    labelMat = []
    fr = open('data/logistic_regression.txt','r') #打开logistics regression 数据集）.txt文件 
                                                                               # 一定要输入正确的数据集文件所在的地址目录
    for line in fr.readlines():                                                #逐行读取
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #因为线性回归化式为 H(x) = W0 + W1*X1 + W2*X2即为 (W0, W1, W2)*(1, X1, X2),
                                                            # 其中 (W0, W1, W2) 即为所求回归系数 W。 为了方便计算, 读出 X1, X2 后要在前面补上一个 1.0
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#分类器的分类（转换）函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))  ##计算 sigmoid 函数

#梯度上升算法，用来计算出最佳回归系数
def gradAscent(dataMatIn, classLabels):
    '''第一个参数是2维数组，每列代表每个不同特征，每行代表每个训练样本
       第二个参数是类别标签，1*100的行向量，为便于计算，将其转换为列向量，即进行转置，并赋值给labelMat
    '''
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix  获得输入数据并将其转换为Numpy矩阵数据类型
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix 获得输入数据并将其转换为Numpy矩阵数据类型
    m, n = shape(dataMatrix)     #shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数
    alpha = 0.001    #步长，向函数增长最快的方向的移动量，即学习率
    maxCycles = 500 #迭代次数
    weights = ones((n, 1))  #生成n行一列的元素为1的矩阵赋给weihts，即回归系数初始化为1

    #循环 maxCycles次, 每次都沿梯度向真实值 labelMat 靠拢
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix multiplication 矩阵相乘 包含了300次的乘积
        error = (labelMat - h)  # vector subtraction 向量减法，计算真实类别与预测类别的差值，h是一个列向量，列向量的元素个数等于样本数，即为100
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix multiplication 矩阵相乘，dataMatrix.transpose()* error 就是梯度f(w)，按照该差值的方向调整回归系数
    return weights

#输出运用梯度上升优化算法后得到的最理想的回归系数的值
def GetResult():
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    plotBestFit(weights)

#画出数据集和Logistic回归最佳拟合直线
def plotBestFit(weights):

    #画点
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 画线
    x = arange(-3.0, 3.0, 0.1)
    y = (0.48 * x + 4.12414) / (0.616)
    # y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()      #显示


if __name__ == '__main__':
    GetResult()