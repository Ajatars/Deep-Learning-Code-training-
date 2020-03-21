import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2017)

#导入数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

plt.plot(x_train, y_train, 'bo')
#把数据转换成tensorflow的tensor形式
x = tf.constant(x_train,name='x')
y = tf.constant(y_train,name='y')

#定义一个线性模型
##varibale是tensorflow下可以修改的值的tensor
w = tf.Variable(initial_value=tf.random_normal(shape=(),seed=2017),dtype=tf.float32,name='weight')
b = tf.Variable(initial_value=0,dtype=tf.float32,name='biase')

#tf.variable_scope()这个函数, 它是用来规定一个变量的区域的, 在这个with语句下定义的所有变量都在同一个变量域当中, 域名就是variable_scope()的参数.
with tf.variable_scope('Linear_Model'):
	y_pred = w*x+b

#开启交互式会话
sess = tf.InteractiveSession()
#初始化
sess.run(tf.global_variables_initializer())#tf.global_variables_initializer()

#将tensor的内容fetch出来
y_pred_numpy = y_pred.eval(session=sess)
plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.legend() #函数主要的作用就是给图加上图例

#优化模型
##定义误差函数
loss = tf.reduce_mean(tf.square(y-y_pred))
print(loss.eval(session=sess))

#梯度下降法
##求导
w_grad,b_grad = tf.gradients(loss,[w,b])
print('w_grad: %.4f' % w_grad.eval(session=sess))
print('b_grad: %.4f' % b_grad.eval(session=sess))

##定义学习率
lr = 1e-2
w_update = w.assign_sub(lr * w_grad)
b_update = b.assign_sub(lr * b_grad)

sess.run([w_update, b_update])

## 更新参数后的模型
y_pred_numpy = y_pred.eval(session=sess)

plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.legend() 

fig = plt.figure() #在plt中绘制一张图片
ax = fig.add_subplot(111)  #新增子图
plt.ion() #显示模式转换为交互（interactive）模式

fig.show()
fig.canvas.draw()
sess.run(tf.global_variables_initializer())

for e in range(10):
	sess.run([w_update, b_update])

	y_pred_numpy = y_pred.eval(session=sess)
	loss_numpy = loss.eval(session=sess)

	ax.clear()
	ax.plot(x_train,y_train,'bo',label='real')
	ax.plot(x_train, y_pred_numpy, 'ro', label='estimated')
	ax.legend()
	fig.canvas.draw()
	plt.pause(0.5)

	print('epoch:{},loss:{}'.format(e,loss_numpy))


plt.plot(x_train, y_train, 'bo', label='real')
plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
plt.legend()

sess.close()