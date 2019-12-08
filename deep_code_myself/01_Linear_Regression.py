import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #내 맥북에서 발생되는 에러를 없애기 위한 코드.

import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)
print('x의 평균값:', mx)
print('y의 평균값:', my)

#기울기 공식의 분모
divisor = sum([(mx - i) ** 2 for i in x])

#기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d

dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)

a = dividend / divisor
b = my - (mx * a)

print("기울기 a=", a)
print("y 절편 b=", b)