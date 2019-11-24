import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import numpy

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:, 0:17]
Y = Data_set[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))