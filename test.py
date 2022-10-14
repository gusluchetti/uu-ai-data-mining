import bestsplit
import gini
import tree
import numpy as np
import time
import os
from pathlib import Path


testing = True


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


def test_tree_grow(printing=True):
    # validate tree with credit data
    nmin = 2
    minleaf = 1
    nfeat = 5
    start_time = time.perf_counter()
    path = os.path.dirname(os.path.abspath(__file__))
    x, y = tree.load_dataset_txt(f'{path}/data/credit.txt')
    root = tree.tree_grow(x, y, nmin, minleaf, nfeat)

    end_time = time.perf_counter()
    if printing:
        tree.traverse(root)

    grow_time = end_time - start_time

    if printing:
        print(grow_time)


def test_tree_pred(printing=False):
    nmin = 20
    minleaf = 5
    start_time_total = time.perf_counter()

    path = os.path.dirname(os.path.abspath(__file__))
    x, y = tree.load_dataset_txt(f'{path}/data/pima.txt')
    nfeat = len(x[0])

    start_time_grow = time.perf_counter()
    root = tree.tree_grow(x, y, nmin, minleaf, nfeat)
    # tree.traverse(root)
    end_time_grow = time.perf_counter()
    predictions = tree.tree_pred(x, root)
    matrix = tree.confusion_matrix(x, y, predictions)

    assert matrix[0][0] < 454
    assert matrix[0][0] > 434
    assert matrix[0][1] < 66
    assert matrix[0][1] > 46
    assert matrix[1][0] < 64
    assert matrix[1][0] > 44
    assert matrix[1][1] < 224
    assert matrix[1][1] > 204

    end_time_total = time.perf_counter()

    if printing:
        print(matrix)
        print("grow time:", end_time_grow-start_time_grow)
        print("total time:", end_time_total - start_time_total)


def test_tree_grow_b(printing=False):
    start_time_total = time.perf_counter()
    nmin = 20
    minleaf = 5
    path = os.path.dirname(os.path.abspath(__file__))
    x, y = tree.load_dataset_txt(f'{path}/data/pima.txt')
    nfeat = len(x[0])
    m = 5
    start_time_grow = time.perf_counter()

    roots = tree.tree_grow_b(x, y, nmin, minleaf, nfeat, m)

    end_time_total = time.perf_counter()

    if printing:
        for i in range(len(roots)):
            print(" ")
            print("Tree number", i)
            tree.traverse(roots[i])

    print("growing time:", end_time_total - start_time_grow)
    print("total time:", end_time_total - start_time_total)

    return x, y, roots


def test_tree_pred_b(x, y, roots, printing=False):
    predictions = tree.tree_pred_b(x, trees=roots)
    matrix = tree.confusion_matrix(x, y, predictions)
    print(matrix)


if testing:
    # test_max_impurity()
    # test_gini()
    # test_bestsplit_num()
    # test_bestsplit_cat()
    # test_tree_grow(printing = False)
    # test_tree_pred(printing = False)
    x, y, roots = test_tree_grow_b(printing=False)
    test_tree_pred_b(x=x, y=y, roots=roots, printing=True)
