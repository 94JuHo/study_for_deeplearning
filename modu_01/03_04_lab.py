import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.)
hypothesis = X * W

#미분에 의한 수식
gradient = tf.reduce_mean((W * X - Y) * X) * 2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#gradient를 임의로 조정할 때 사용할 수 있
#Get gradients
gvs = optimizer.compute_gradients(cost)
#Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)