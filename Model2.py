import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 将之前的graph清除
tf.reset_default_graph()

# 定义一个多变量函数
# 定义参数
w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

# 打印函数式子
f_des = 'y={:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2])

print(f_des)

# 画出这个函数的曲线
x_sample = np.arange(-3, 3.1, 0.1) #创建等差数组
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3

# plt.plot(x_sample,y_sample,label='real curve')
# plt.legend()

#多项式回归问题转换为线性回归问题
x_train = np.stack([x_sample ** i for i in range(1,4)],axis=1) #对每个元素增加一维
x_train = tf.constant(x_train,dtype=tf.float32,name='x_train')
y_train = tf.constant(y_sample,dtype=tf.float32,name='y_train')

#构造线性模型
w = tf.Variable(initial_value=tf.random_normal(shape=(3,1)),dtype=tf.float32,name='weights')
b = tf.Variable(initial_value=0,dtype=tf.float32,name='bias')

def multi_linear(x):
    return tf.squeeze(tf.matmul(x, w) + b) #矩阵x,w乘法 + b 然后默认删除所有为1的维度

y_ = multi_linear(x_train)

#开启交互式会话
sess = tf.InteractiveSession()
#初始化
sess.run(tf.global_variables_initializer())


x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)

#画出模型输出的结果和真实结果的对比
# plt.plot(x_train_value[:,0], y_pred_value, label='fitting curve', color='r')
# plt.plot(x_train_value[:,0], y_train_value, label='real curve', color='b')
# plt.legend()


#定义损失函数
loss  = tf.reduce_mean(tf.square(y_train-y_))
loss_numpy = sess.run(loss)
print(loss_numpy)

#求导
w_grad,b_grad = tf.gradients(loss,[w,b])

print(w_grad.eval(session=sess))
print(b_grad.eval(session=sess))

#梯度下降法
lr = 1e-3 #学习率

w_update = w.assign_sub(lr * w_grad)
b_update = b.assign_sub(lr * b_grad)

sess.run([w_update,b_update])

x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)
loss_numpy =  loss.eval(session=sess)

plt.plot(x_train_value[:,0], y_pred_value, label='fitting curve', color='r')
plt.plot(x_train_value[:,0], y_train_value, label='real curve', color='b')
plt.legend()
plt.title('loss: %.4f' % loss_numpy)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

sess.run(tf.global_variables_initializer())

for e in range(100):
	sess.run([w_update,b_update])

	x_train_value = x_train.eval(session=sess)
	y_train_value = y_train.eval(session=sess)
	y_pred_value = y_.eval(session=sess)
	loss_numpy = loss.eval(session=sess)

	ax.clear()
	ax.plot(x_train_value[:,0],y_pred_value,label='fitting curve',color='r')
	ax.plot(x_train_value[:,0],y_train_value,label='real curve',color='b')
	ax.legend()

	fig.canvas.draw()
	plt.pause(0.1)

	if (e+1)%20 == 0:
		print('epoch:{},loss:{}'.format(e+1,loss_numpy))