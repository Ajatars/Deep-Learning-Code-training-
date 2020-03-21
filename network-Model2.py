import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
tf.set_random_seed(2017)

def plot_decision_boundary(model,x,y):
	x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
	y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
	h = 0.01

	xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max))

	Z = model(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(x[:,0],x[:,1],c=np.squeeze(y),cmap=plt.cm.Spectral)
	plt.show()


np.random.seed(1)
m = 400
N = int(m/2)
D = 2
x = np.zeros((m,D))
y = np.zeros((m,1),dtype='uint8')
a = 4

for j in range(2):
	ix = range(N*j,N*(j+1))
	t = np.linspace(j*3.12,(j+1)*3.12, N) +np.random.randn(N)*0.2
	r = a*np.sin(4*t) + np.random.randn(N)*0.2
	x[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
	y[ix] = j

x = tf.constant(x, dtype=tf.float32, name='x')
y = tf.constant(y, dtype=tf.float32, name='y')

w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(2, 1), dtype=tf.float32, name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), dtype=tf.float32, name='bias')

def logistic_model(x):
	logit = tf.matmul(x, w) + b
	return tf.sigmoid(logit)

model_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='logistic_input')
logistic_output = logistic_model(model_input)

def hidden_layer(layer_input,output_depth,scope='hidden_layer',reuse=None):
	input_depth = layer_input.get_shape()[-1]
	with tf.variable_scope(scope,reuse=reuse):
		w = tf.get_variable(initializer=tf.random_normal_initializer(),shape=(input_depth,output_depth),name='weights')
		b = tf.get_variable(initializer=tf.zeros_initializer(),shape=(output_depth),name='bais')
		net = tf.matmul(layer_input,w)+b

		return net

def DNN(x,output_depths,scop='DNN',reuse=None):
	net = x
	for i,output_depth in enumerate(output_depths):
		net = hidden_layer(net,output_depth,scope='layer%d'%i,reuse=reuse)
		net = tf.tanh(net)
	net = hidden_layer(net,1,scope='classification',reuse=reuse)
	net = tf.sigmoid(net)

	return net

dnn = DNN(x,[10,10,10])

loss_dnn = tf.losses.log_loss(predictions=dnn,labels=y)
lr = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_dnn)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for e in range(20000):
    sess.run(train_op)
    if (e + 1) % 1000 == 0:
        loss_numpy = loss_dnn.eval(session=sess)
        print('Epoch {}: Loss: {}'.format(e + 1, loss_numpy))

dnn_out = DNN(model_input, [10, 10, 10], reuse=True)

def plot_dnn(input_data):
    y_pred_numpy = sess.run(dnn_out, feed_dict={model_input: input_data})
    out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
    return np.squeeze(out)

plot_decision_boundary(plot_dnn, x.eval(session=sess), y.eval(session=sess))
plt.title('4 layer network')