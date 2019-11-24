import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드

import numpy as np

ab = [3, 76]

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

#RMSE function
def rmse(p, a):
    return np.sqrt(((p-a) ** 2).mean())

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict_result[i]))

print("rmse 최종값: " + str(rmse_val(predict_result, y)))