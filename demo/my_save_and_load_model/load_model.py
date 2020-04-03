__author__ = 'liuwei'

import tensorflow as tf
import numpy as np

h_est = tf.Variable(tf.random_uniform([1], -1, 1))     #只定义，没有初始化
v_est = tf.Variable(tf.random_uniform([1], -1, 1))


saver = tf.train.Saver()                      #saver类

path = './final_model'                        #要恢复的checkpoint路径

with tf.Session() as sess:
    saver.restore(sess, path)                 #恢复参数

    print(sess.run(h_est), sess.run(v_est))