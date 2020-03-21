import tensorflow.compat.v1 as tf 
import matplotlib.pyplot as plt
import numpy as np
import time

tf.disable_v2_behavior()

#从 data.txt中读入点
with open('data/logistic_regression.txt','r') as f:
	data_list = [i.strip().split() for i in f.readlines()]
	data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max,i[1]/x1_max,i[2]) for i in data]

x0 = list(filter(lambda x:x[-1] == 0.0,data)) #选择第一类的点
x1 = list(filter(lambda x:x[-1] == 1.0,data)) #选择第二类的点

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

# plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
# plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
# plt.legend(loc='best')

#定义一个线性模型
##varibale是tensorflow下可以修改的值的tensor
np_data = np.array(data,dtype='float32') #转换成numpy array
x_data = tf.constant(np_data[:, 0:2], name='x')
y_data = tf.expand_dims(tf.constant(np_data[:,-1]),axis=-1) #增加维度


w = tf.get_variable(initializer=tf.random_normal_initializer(seed=2017), shape=(2, 1), name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), name='bias') #生成初始化为0的张量的初始化器

# 使用 tf.sigmoid 将结果映射到 [0, 1] 区间
def logistic_regression(x):
	return tf.sigmoid(tf.matmul(x,w)+b)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#初始分类效果
w_numpy = w.eval(session=sess)
b_numpy = b.eval(session=sess)

w0 = w_numpy[0]
w1 = w_numpy[1]
b0 = b_numpy[0]

plot_x = np.arange(0.2,1,0.01)
plot_y = (-w0 * plot_x -b0)/w1

# plt.plot(plot_x, plot_y, 'g', label='cutting line')
# plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
# plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
# plt.legend(loc='best')

#优化模型
##定义损失函数
def binary_loss(y_pred,y):
	logit = tf.reduce_mean(y*tf.log(y_pred) + (1-y) * tf.log(1-y_pred))
	return -logit

y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred,y_data)

print(loss.eval(session=sess))

## 梯度计算及参数跟新
### 求导
w_grad,b_grad = tf.gradients(loss,[w,b])

lr = 0.1
w_update = w.assign_sub(lr*w_grad)
b_update = b.assign_sub(lr*b_grad)

### 更新参数
sess.run([w_update,b_update])
### 查看更新后的loss
print(loss.eval(session=sess))


#使用tensorflow内置loss
##Logistic 回归的二分类 loss 在 tensorflow 中是 tf.losses.log_loss
loss1 = tf.losses.log_loss(predictions=y_pred,labels=y_data)
### 从tf.train中定义一个优化方法
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=lr)
### 利用优化方法去优化一个损失函数，得到这个op 就是我们想要的
train_op1 = optimizer1.minimize(loss1)

sess.run(tf.global_variables_initializer())
start = time.time()
for e in range(1000):
	if(e+1)%200 ==0:
		y_true_label = y_data.eval(session=sess)
		y_pred_numpy = y_pred.eval(session=sess)
		#np.greater_equal(x1, x2 [,y]) 比较运算函数 y = x1 >= x2  astype数据转换
		y_pred_label = np.greater_equal(y_pred_numpy,0.5).astype(np.float32)
		#求取均值
		accuracy = np.mean(y_pred_label == y_true_label)
		loss_numpy = loss.eval(session=sess)
		print('Epoch %d, Loss: %.4f, Acc: %.4f' % (e + 1, loss_numpy, accuracy))

print('Tensorflow_GD cost time: %.4f' % (time.time() - start))