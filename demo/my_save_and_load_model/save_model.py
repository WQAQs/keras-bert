__author__ = 'liuwei'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

h = 1
v = -2

#prepare data
x_train = np.linspace(-2, 4, 201)                        #x样本
noise = np.random.randn(*x_train.shape) * 0.4            #噪音
y_train = (x_train - h) ** 2 + v + noise                 #y样本

n = x_train.shape[0]

x_train = np.reshape(x_train, (n, 1))                    #重塑
y_train = np.reshape(y_train, (n, 1))

#画出产生的数据的形状
'''
plt.rcParams['figure.figsize'] = (10, 6)
plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()
'''
#create variable
X = tf.placeholder(tf.float32, [1])                      #两个占位符，x和y
Y = tf.placeholder(tf.float32, [1])

h_est = tf.Variable(tf.random_uniform([1], -1, 1))       #定义需要训练的参数，在saver之前定义
v_est = tf.Variable(tf.random_uniform([1], -1, 1))

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)                                 #保存模型参数的saver

value = (X - h_est) ** 2 + v_est                         #拟合的曲线

loss = tf.reduce_mean(tf.square(value - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(2):                             #100个epoch
        for (x, y) in zip(x_train, y_train):

            sess.run(optimizer, feed_dict={X: x, Y: y})
        #保存checkpoint
        saver.save(sess, 'model/model.ckpt', global_step=epoch)

    #saver the final model
    saver.save(sess, 'model/model.ckpt')                    #最后一个epoch对应的checkpoint
    h_ = sess.run(h_est)
    v_ = sess.run(v_est)

    print(h_, v_)