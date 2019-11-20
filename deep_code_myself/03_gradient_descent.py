import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import tensorflow as tf

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

a = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float64, seed=0))

y = a * x_data + b

rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data)))

learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print(step, sess.run(rmse), sess.run(a), sess.run(b))