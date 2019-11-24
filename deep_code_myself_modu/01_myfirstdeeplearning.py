import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import tensorflow as tf
import numpy


seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:,0:17]
Y = Data_set[:, 17]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, input_dim=17, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
