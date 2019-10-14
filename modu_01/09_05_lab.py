import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
    b2 = tf.Variable(tf.random_normal([10]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("biases4", b4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log( 1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
cost_summ = tf.summary.scalar("cost", cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    for step in range(10001):
        _, s = sess.run([train, summary], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s, global_step=step)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run([W1, W2, W3, W4]))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y:y_data})
    print("\nH: ", h, "\n C:",c,"\n Acc:", a)
