import numpy as np


def bestsplit(x, y):
    length = len(x)-1
    x_sorted = np.sort(np.unique(x))
    x_splitpoints = (x_sorted[0:(length-2)]+x_sorted[1:(length-1)])/2
    print(x_sorted, x_splitpoints)
    return True


# based on income data
x = [28, 32, 24, 27, 32, 30, 58, 52, 40, 28]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

assert len(x) == len(y)
bestsplit(x, y)
