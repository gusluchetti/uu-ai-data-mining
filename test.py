from gini import gini_impurity
from bestsplit import bestsplit
import numpy as np


def test_gini():
    array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    expected_gini = 0.23140495867768596
    assert gini_impurity(array) == expected_gini


def test_bestsplit():
    # values based on income data
    x = np.array([28, 32, 24, 27, 32, 30, 58, 52, 40, 28])
    y = np.array([0,   0,  0,  0,  0,  1,  1,  1,  1,  1])
    assert len(x) == len(y)
    expected_best = 36
    assert bestsplit(x, y) == expected_best
