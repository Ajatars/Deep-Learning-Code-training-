import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

tf.set_random_seed(2017)

def plot_decision_boundary(model,x,y):
	#找到x,y的最大值和最小值，并在周围填充一个像素
	x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
	y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
	h = 0.01
	#构建一个宽度为`h`的网络
	'''
	numpy.meshgrid() 生成网格点坐标矩阵
	np.arange() 返回一个有终点和起点的固定步长的排列
	'''
	xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
	#计算模型在网格上的所有点的输出值
	'''
	np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
	np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
	ravel() 将多维数组降位一维
	shape 是查看数据有多少行多少列
	reshape()是数组array中的方法，作用是将数据重新组织,更改数组维数
	squeeze()只能对维数为1的维度降维
	'''
	Z = model(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)

	#画图显示
	'''
	plt.contour()和contourf() 都是画三维等高线图的，不同点在于contour() 是绘制轮廓线，contourf()会填充轮廓。
	plt.cm.Spectral 实现的功能是给label为1的点一种颜色，给label为0的点另一种颜色。
	plt.scatter() 绘制散点图的数据点
	'''
	plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(x[:,0],x[:,1],c=np.squeeze(y),cmap=plt.cm.Spectral)
	plt.show()

#数据集
np.random.seed(1)
m = 400 #样本数量
N = int(m/2) #每一类的点的个数
D = 2 #维度
x = np.zeros((m,D))  #np.zeros 返回来一个给定形状和类型的用0填充的数组；
y = np.zeros((m,1),dtype='uint8') #label 向量，0表示红色，1表示蓝色
a = 4

for j in range(2):
	ix = range(N*j,N*(j+1))
	t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 #theta
	r = a*np.sin(4*t) + np.random.randn(N)*0.2 #radius
	x[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
	y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral)
plt.show()

#尝试用logistic回归解决
x = tf.constant(x,dtype=tf.float32,name='x')
y = tf.constant(y,dtype=tf.float32,name='y')

#定义模型
w = tf.get_variable(initializer=tf.random_normal_initializer(),shape=(2,1),dtype=tf.float32,name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), dtype=tf.float32, name='bias')

def logistic_model(x):
	logit = tf.matmul(x,w) + b
	return tf.sigmoid(logit)

y_ = logistic_model(x)

#构造训练
loss = tf.losses.log_loss(predictions=y_,labels=y)

lr = 1e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

#执行训练
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for e in range(1000):
	sess.run(train_op)
	if(e+1)%100 ==0:
		loss_numpy = loss.eval(session=sess)
		print('Epoch %d: Loss: %.12f' % (e + 1, loss_numpy)) #可以看到loss没有下降

'''placeholder()函数是在神经网络构建graph的时候在模型中的占位，
此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，
在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。'''
model_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='logistic_input')
logistic_output = logistic_model(model_input)

def plot_logistic(x_data):
    y_pred_numpy = sess.run(logistic_output, feed_dict={model_input: x_data})
    #np.greater() >
    out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
    return np.squeeze(out)

plot_decision_boundary(plot_logistic, x.eval(session=sess), y.eval(session=sess))

#深度神经网络
##构建第一个隐藏层
with tf.variable_scope('layerl'):
	#构建参数weight,stddev：一个 python 标量或一个标量张量.要生成的随机值的标准偏差.
	w1 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01),shape=(2,4), name='weights1')
	#构建参数bias
	b1 = tf.get_variable(initializer=tf.zeros_initializer(),shape=(4),name='bias1')

##构建第二个隐藏层
with tf.variable_scope('layer2'):
	w2 = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.01), shape=(4, 1), name='weights2')
	b2 = tf.get_variable(initializer=tf.zeros_initializer(),shape=(1),name='bias2')

def two_network(nn_input):
	with tf.variable_scope('two_network'):
		#第一个隐藏层
		net = tf.matmul(nn_input,w1)+b1
		# tanh激活层
		net = tf.tanh(net)
		# 第二个隐藏层
		net = tf.matmul(net,w2) + b2

		# 经过sigmoid得到输出
		return tf.sigmoid(net)

net = two_network(x)

# 构建神经网络的训练过程
#使用tensorflow内置loss
loss_two = tf.losses.log_loss(predictions=net,labels=y,scope='loss_two')
lr = 1
### 从tf.train中定义一个优化方法
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
### 利用优化方法去优化一个损失函数，得到这个op 就是我们想要的
train_op = optimizer.minimize(loss=loss_two,var_list=[w1,w2,b1,b2])

### 模型的保存
'''
在我们开始训练之前, 我们先思考一下,
在之前的过程中, 当模型的参数经过训练之后,
模型的效果得以显示, 而模型的参数却没有得到保存,
么下次我们希望得到一个比较好的结果的时候,
又必须得重新训练, 这是令人难以接受的.
因此, 我们需要将模型保存到本地,
 并且需要一种正确的方式将模型读入到内存中来.

 Tensorflow提供了tf.train.Saver类来管理模型的保存与加载
'''
saver = tf.train.Saver()

# 我们训练10000次

sess.run(tf.global_variables_initializer())

for e in range(10000):
	sess.run(train_op)
	if (e+1)%1000 == 0:
		loss_numpy = loss_two.eval(session=sess)
		print('Epoch {}: Loss: {}'.format(e + 1, loss_numpy))
	if (e+1)%5000 == 0:
		# `sess`参数表示开启模型的`session`, 必选参数
        # `save_path`参数表示模型保存的路径, 必须要以`.ckpt`结尾
        # `global_step`参数表示模型当前训练的步数, 可以用来标记不同阶段的模型
		saver.save(sess=sess, save_path='First_Save/model.ckpt', global_step=(e + 1))
'''
model.ckpt-5000和model.ckpt-10000都是保存好的模型, 
而他们又都有.data-00000-of-00001, .meta和.index三个文件. 
这是由于tensorflow在保存的过程中同时保存了模型的定义和模型参数的值, 
然后又分开不同的文件存放.
'''
#查看模型训练效果
nn_out = two_network(model_input)

def plot_network(input_data):
	y_pred_numpy = sess.run(nn_out,feed_dict={model_input:input_data})
	out = np.greater(y_pred_numpy,0.5).astype(np.float32)
	return np.squeeze(out)

plot_decision_boundary(plot_network, x.eval(session=sess), y.eval(session=sess))
plt.title('2 layer network')

#关闭当前会话
sess.close()

sess = tf.InteractiveSession()

# 用try语句打印一下`w1`的值
try:
    print(w1.eval(session=sess))
except tf.errors.FailedPreconditionError as e:
    print(e.message)

#模型恢复
## 恢复模型结构
saver = tf.train.import_meta_graph('First_Save/model.ckpt-10000.meta')

## 恢复模型参数
saver.restore(sess,'First_Save/model.ckpt-10000')

#打印w1
print(w1.eval(session=sess))
#重新可视化分类结果
plot_decision_boundary(plot_network, x.eval(session=sess), y.eval(session=sess))
plt.title('2 layer network')