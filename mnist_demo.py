import tensorflow as tf
import numpy as np
import cv2
import struct
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

batch_size = 100
n_batch = 600

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001,dtype=tf.float32)

Weight_l1 = tf.Variable(tf.truncated_normal([784, 100],stddev=0.1))
Bias_l1 = tf.Variable(tf.zeros([100])+0.1)
L1 = tf.nn.tanh(tf.matmul(x, Weight_l1) + Bias_l1)
L1_drop = tf.nn.dropout(L1,keep_prob)

Weight_l2 =  tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
Bias_l2 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.matmul(L1_drop,Weight_l2) + Bias_l2

#loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels,keep_prob:1})
        print(str(acc))
