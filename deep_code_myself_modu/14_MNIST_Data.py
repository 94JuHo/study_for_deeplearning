#-*-coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

from keras.datasets import mnist
from keras.utils import np_utils

import sys
import tensorflow as tf
import numpy

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

print("학습셋 이미지 수 : %d" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d" % (X_test.shape[0]))

import matplotlib.pyplot as plt

plt.imshow(X_train[1], cmap='Greys')
plt.show()

for x in X_train[1]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

print('class : %d' %(Y_class_train[1]))

Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

print(Y_train[1])