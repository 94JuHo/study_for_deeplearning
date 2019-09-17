import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import tensorflow as tf

#trainable variable이다. 학습과정에서 변경될 수 있는 값이다.
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]
#placeholder를 사용해서 출력단에서 값 입력받기
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W,b를 모르기 때문에 랜덤한 값을 만든다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our hypothesis XW+b
# hypothesis = x_train * W + b
hypothesis = X * W + b

#cost/loss function
#cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#reduce_mean은 tensor가 주어지면 그것의 평균을 내주는 것임

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #variable을 실행하기전에는 무조건 이 함수를 통해 초기화시켜줘야함

for step in range(4001):
    # sess.run(train)
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1, 2, 3, 4, 5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        #print(step, sess.run(cost), sess.run(W), sess.run(b))
        print(step, cost_val, W_val, b_val)