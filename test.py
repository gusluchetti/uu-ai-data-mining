import bestsplit
import gini
import tree
import numpy as np


def test_max_impurity():
    # asserting odd char array with max impurity
    array = np.array(['A', 'B', 'C'])
    expected_gini = gini.get_max_impurity(array)
    assert gini.gini_impurity(array) == expected_gini


def test_gini():
    # asserting binary array with given answer
    array = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])  # 4 / 7
    expected_gini = 0.23140495867768596 * 2
    assert gini.gini_impurity(array) == expected_gini

    # asserting char array with max impurity
    array = np.array([0, 1])
    expected_gini = 1/2 * 1/2 * 2  # 1/2 * 1/2
    assert gini.gini_impurity(array) == expected_gini


def test_bestsplit_num():
    # values based on income data
    x = np.array([28, 32, 24, 27, 32, 30, 58, 52, 40, 28])
    y = np.array([0,   0,  0,  0,  0,  1,  1,  1,  1,  1])
    assert len(x) == len(y)
    expected_best = 36
    assert bestsplit.get_bestsplit(x, y) == expected_best

    # values based on hw set 1 - ex 1
    x = np.array([2, 2, 3, 4, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 2, 2, 2])
    assert len(x) == len(y)
    expected_best = 6.5
    assert bestsplit.get_bestsplit(x, y) == expected_best


def test_bestsplit_cat():
    # values based on hw set 1 - ex 2
    x = np.array(['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])
    y = np.array([0,    0,   1,   0,   1,   1,   1,   0,    0])
    assert len(x) == len(y)
    expected_best = ['c']
    assert bestsplit.get_bestsplit_cat(x, y) == expected_best


def test_tree_grow():
    # validate tree with credit data
    nmin = 2
    minleaf = 1
    nfeat = 5

    credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
    print(credit_data)
    # df = tree.load_dataset('data.csv')
    last = len(credit_data[0]) - 1
    x = credit_data[:, 0:last]
    y = credit_data[:, last]

    root = tree.tree_grow(x, y, nmin, minleaf, nfeat)
    tree.traverse(root)
