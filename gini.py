import numpy as np


# get uniques
# count frequency of each unique
# compute gini
# profit?
def gini_impurity(array: np.ndarray):
    # P(i) * (1 - P(i))
    """ updated version of gini impurity to handle separation
    between a number N of different classes
    classe could either be numbers or chars

    args: np array of N classes
    returns: weighted gini_index of node
    """
    print(array)
    length = len(array)
    if length == 1:
        return 0.0
    count_0 = np.count_nonzero(array == 0)
    p0 = count_0/length
    p1 = 1 - count_0/length
    gini = p0 * p1
    print(f"Gini = {gini}")
    return gini
