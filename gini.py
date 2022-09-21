import numpy as np


def gini_impurity(array):
    print(array)
    # P(i) * (1 - P(i))
    length = len(array)
    if length == 1:
        return 0.0
    count_0 = np.count_nonzero(array == 0)
    p0 = count_0/length
    p1 = 1 - count_0/length
    gini = p0 * p1
    print(f"Gini = {gini}")
    return gini
