import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.disable_v2_behavior()

tf.set_random_seed(2017) #设置图级随机seed。

#导入mnist数据集
##read_data_sets的一个参数one_hot,将一个数值n映射到一个向量，这个向量的第n个元素是1，其他都是0
mnist = input_data.read_data_sets(r'data\MNIST_data',one_hot=True)

#数据集分成两个部分，训练和测试，分开来是为了观察模型在完全没见过的数据上的表现，从而体现泛化能力
train_set = mnist.train
test_set = mnist.test

#把父图分为6*2个子图，
fig,axes = plt.subplots(ncols=6,nrows=2)
##tight_layout会自动调整子图参数，使之填充整个图像区域
plt.tight_layout(w_pad=-2.0,h_pad=-8.0)

#调用next_batch方法来一次性获取12个样本
## shuffle参数，表示是否打乱样本间的顺序
images,labels = train_set.next_batch(12,shuffle=False)

for ind,(image,label) in enumerate(zip(images,labels)):
	#image 是一个784维的向量，是图片进行拉伸的产生的，这里reshape回去
	image = image.reshape((28,28))

	# label是一个10维的向量，哪个下标处值为1，说明数字是几
	##argmax()参数返回的是沿轴axis最大值的索引值
	label = label.argmax()

	row = ind//6
	col = ind%6
	#imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。
	axes[row][col].imshow(image,cmap='gray') #灰度图
	#axis是用来设置具体某一个坐标轴的属性的
	axes[row][col].axis('off')
	axes[row][col].set_title('%d'%label)

#定义深度网络结构
def hidden_layer(layer_input,output_depth,scope='hidden_layer',reuse=None):
	input_depth = layer_input.get_shape()[-1]
	with tf.variable_scope(scope,reuse=reuse):
		#初始化方法是truncated_normal 产生截断正态分布随机数
		w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),shape=(input_depth,output_depth),name='weights')
		#用0.1对偏置进行初始化
		b = tf.get_variable(initializer=tf.constant_initializer(0.1),shape=(output_depth),name='bias')

		net = tf.matmul(layer_input,w) + b

		return net

def DNN(x,output_depths,scope='DNN',reuse=None):
	net = x
	for i,output_depth in enumerate(output_depths):
		net = hidden_layer(net,output_depth,scope='layer%d'%i,reuse=reuse)
		#激活函数
		net = tf.nn.relu(net)
	#数字分为0,1...，9 所以输出一个10维的向量
	net = hidden_layer(net,10,scope='classification',reuse=reuse)

	return net

#占位符
#shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
input_ph = tf.placeholder(shape=(None,784),dtype=tf.float32)
label_ph = tf.placeholder(shape=(None,10),dtype=tf.int64)

#构造一个4层的神经网络，隐藏节点数为:400,200,100,10
dnn = DNN(input_ph,[400,200,100])

#交叉熵计算损失函数
loss = tf.losses.softmax_cross_entropy(logits=dnn,onehot_labels=label_ph)

#下面定义正确率
##equal()对比两个矩阵或者向量相等的元素，argmax()，返回最大值的索引值，cast()，把数据格式转换成dtype，reduce_mean求平均值
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn,axis=-1),tf.argmax(label_ph,axis=-1)),dtype=tf.float32))

lr = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()

# 我们训练20000次

batch_size = 64

sess.run(tf.global_variables_initializer())

for e in range(20000):
    # 获取 batch_size个训练样本
    images, labels = train_set.next_batch(batch_size)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
    if e % 1000 == 999:
        # 获取 batch_size 个测试样本
        test_imgs, test_labels = test_set.next_batch(batch_size)
        # 计算在当前样本上的训练以及测试样本的损失值和正确率
        loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: images, label_ph: labels})
        loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: test_imgs, label_ph: test_labels})
        print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f} test_acc: {:.6f}'.format(e + 1, loss_train, acc_train, loss_test, acc_test))

print('Train Done!')
print('-'*30)


# 计算所有训练样本的损失值以及正确率
train_loss = []
train_acc = []
for _ in range(train_set.num_examples // 100):
    image, label = train_set.next_batch(100)
    loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    train_loss.append(loss_train)
    train_acc.append(acc_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))
print('Train accuracy: {:.6f}'.format(np.array(train_acc).mean()))

# 计算所有测试样本的损失值以及正确率
test_loss = []
test_acc = []
for _ in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    test_loss.append(loss_test)
    test_acc.append(acc_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))
print('Test accuracy: {:.6f}'.format(np.array(test_acc).mean()))

sess.close()

#使用summary
##重置计算图
tf.reset_default_graph()

##重新定义占位符
input_ph = tf.placeholder(shape=(None,784),dtype=tf.float32)
label_ph = tf.placeholder(shape=(None,10),dtype=tf.int64)

##重新构建前向神经网络, 为了简化代码, 我们在构造一个隐藏层以及它的参数的函数内部构造tf.summary
###构造权重,用`truncated_normal`初始化
def weight_variable(shape):
	init = tf.truncated_normal(shape=shape,stddev=0.1)
	return tf.Variable(init)

###构造偏置，用`0.1`初始化
def bias_variable(shape):
	init = tf.constant(0.1,shape=shape)
	return tf.Variable(init)

###构造添加`variable`的`summary`的函数
def variable_summaries(var):
	with tf.name_scope('summaries'):
		#计算平均值
		mean = tf.reduce_mean(var)
		#将平均值放到`summary`
		tf.summary.scalar('mean',mean)

		#计算标准差
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		#将标准差添加到`summary`中
		tf.summary.scalar('stddev',stddev)

		#添加最大值和最小值到`summary`
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.scalar('min',tf.reduce_min(var))

		#添加这个变量分布情况`summary`
		tf.summary.histogram('histogram',var)

#构造一个隐藏层
def hidden_layer(x, output_dim, scope='hidden_layer',act=tf.nn.relu, reuse=None):
	#获取输入的`depth`
	input_dim = x.get_shape().as_list()[-1]

	with tf.name_scope(scope):
		with tf.name_scope('weight'):
			#构造`weight`
			weight = weight_variable([input_dim,output_dim])
			#添加`weight`的`summary`
			variable_summaries(weight)

		with tf.name_scope('bias'):
			# 构造`bias`
			bias = bias_variable([output_dim])
			# 添加`bias`的`summary`
			variable_summaries(bias)

		with tf.name_scope('linear'):
			#计算`xw+b`
			preact = tf.matmul(x,weight) + bias
			tf.summary.histogram('pre_activation',preact)

		#经过激活层`act`
		output = act(preact)
		#添加激活后输出到分布情况到`summary`
		tf.summary.histogram('output',output)
		return output

# 构造深度神经网络
def DNN(x, output_depths, scope='DNN_with_sums', reuse=None):
    with tf.name_scope(scope):
        net = x
        for i, output_depth in enumerate(output_depths):
            net = hidden_layer(net, output_depth, scope='hidden%d' % (i + 1), reuse=reuse)
        # 最后有一个分类层
        net = hidden_layer(net, 10, scope='classification', act=tf.identity, reuse=reuse)
        return net

dnn_with_sums = DNN(input_ph, [400, 200, 100])

# 重新定义`loss`, `acc`, `train_op`
with tf.name_scope('cross_entropy'):
    loss = tf.losses.softmax_cross_entropy(logits=dnn_with_sums, onehot_labels=label_ph)
    tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('accuracy'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn_with_sums, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))
    tf.summary.scalar('accuracy', acc)

with tf.name_scope('train'):
    lr = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)


merged = tf.summary.merge_all()
sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('test_summary/train',sess.graph)
test_writer = tf.summary.FileWriter('test_summary/test',sess.graph)

batch_size = 64
sess.run(tf.global_variables_initializer())

for e in range(20000):
	images,labels = train_set.next_batch(batch_size)
	sess.run(train_op,feed_dict={input_ph:images,label_ph:labels})

	if e%1000 == 999:
		test_imgs,test_labels = test_set.next_batch(batch_size)
		#获取train的数据的summaries 以及 loss，acc的信息
		sum_train,loss_train,acc_train = sess.run([merged,loss,acc],feed_dict={input_ph: images, label_ph: labels})
		#将 train 的summaries 写入到 train_writer
		train_writer.add_summary(sum_train,e)
		#获取 test数据的summaries 以及 loss ，acc的信息
		sum_test, loss_test, acc_test = sess.run([merged, loss, acc], feed_dict={input_ph: test_imgs, label_ph: test_labels})
		# 将 test 的summaries写入到test_writer中
		test_writer.add_summary(sum_test, e)
		print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f} test_acc: {:.6f}'.format(e + 1, loss_train, acc_train, loss_test, acc_test))

# 关闭读写器
train_writer.close()
test_writer.close()

print('Train Done!')
print('-'*30)

# 计算所有训练样本的损失值以及正确率
train_loss = []
train_acc = []
for _ in range(train_set.num_examples // 100):
    image, label = train_set.next_batch(100)
    loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    train_loss.append(loss_train)
    train_acc.append(acc_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))
print('Train accuracy: {:.6f}'.format(np.array(train_acc).mean()))

# 计算所有测试样本的损失值以及正确率
test_loss = []
test_acc = []
for _ in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    test_loss.append(loss_test)
    test_acc.append(acc_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))
print('Test accuracy: {:.6f}'.format(np.array(test_acc).mean()))