import numpy as np


def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)

divisor = sum([(mx - i) ** 2 for i in x])
dividend = top(x, mx, y, my)

a = dividend / divisor
b = my - (mx * a)

print(a, b)
