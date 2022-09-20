import numpy as np


def gini_impurity(array):
    # P(i) * (1 - P(i))
    length = len(array)
    count_0 = np.count_nonzero(array == 0)
    p0 = count_0/length
    p1 = 1 - count_0/length
    # print(f"p0 = {p0}; p1 = {p1};")
    return p0*p1


array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
expected_gini = 0.23140495867768596
gini = gini_impurity(array)
assert gini == expected_gini
