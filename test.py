import bestsplit
import gini
import numpy as np


# test suite for different functions being used
def test_gini():
    # asserting char array with max impurity
    array = np.array([0, 1])
    expected_gini = 0.25  # 1/2 * 1/2
    assert gini.gini_impurity(array) == expected_gini

    # asserting binary array with given answer
    array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_gini = 0.23140495867768596
    assert gini.gini_impurity(array) == expected_gini

    # asserting odd char array with max impurity
    array = np.array(['A', 'B', 'C'])
    expected_gini = 1/3 * 1/3 * 1/3
    assert gini.gini_impurity(array) == expected_gini


def test_bestsplit_num():
    # values based on income data
    x = np.array([28, 32, 24, 27, 32, 30, 58, 52, 40, 28])
    y = np.array([0,   0,  0,  0,  0,  1,  1,  1,  1,  1])
    assert len(x) == len(y)
    expected_best = 36
    assert bestsplit.get_bestsplit(x, y) == expected_best


def test_bestsplit_cat():
    # values based on homework set 1
    x = np.array(['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])
    y = np.array([0,    0,   1,   0,   1,   1,   1,   0,    0])
    assert len(x) == len(y)
    expected_best = ['c']
    assert bestsplit.get_bestsplit_cat(x, y) == expected_best
