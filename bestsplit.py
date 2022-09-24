import numpy as np
from math import pow
from gini import gini_impurity, get_max_impurity
from itertools import combinations


# TODO: any way of getting the setup for both best splits into functions?
def get_impr_red(left, right, max_impr):
    """ Calculate impurity reduction based on two leaf nodes and max impurity """
    gini_left = gini_impurity(left)
    gini_right = gini_impurity(right)
    freq_l = len(left)/(len(left)+len(right))
    freq_r = 1 - freq_l
    print(gini_left, gini_right, freq_l, freq_r)
    return max_impr - ((freq_l*gini_left) + ((freq_r)*gini_right))


def get_bestsplit(x, y):
    """ Get best split for an array of numerical features and class labels """
    max_impr = get_max_impurity(np.unique(y))
    best_red = float('-inf')
    best_split = None
    x_sorted = np.sort(np.unique(x))
    x_splitpoints = (
        x_sorted[0:(len(x)-3)] + x_sorted[1:(len(x)-2)]
    )/2
    print(f"Splitpoints -> {x_splitpoints}")

    for split in x_splitpoints:
        print(f"Split -> {split}")
        left, right = y[x < split], y[x >= split]

        impr_red = get_impr_red(left, right, max_impr)
        print(f"Cur. Impurity Reduction = {impr_red}\n")

        if impr_red > best_red:
            best_red = impr_red
            best_split = split
        print(f"Best split {best_split} has {best_red} impurity reduction\n")

    return best_split


def get_bestsplit_cat(x, y):
    """ Get best split for an array of categorical features and class labels """
    max_impr = get_max_impurity(np.unique(y))
    best_red = float('-inf')
    best_split = None
    x_uniques = np.unique(x)
    length = x_uniques.size
    num_splits = pow(2, length - 1) - 1  # 2^L-1 -1 (L = unique cat vars)
    all_splits = []

    x_as_string = ""
    x_as_string = x_as_string.join(x_uniques)
    # get all possible splits
    for i in range(length):
        for c in combinations(x_as_string, i+1):
            all_splits.append(list(c))
            num_splits -= 1
            if num_splits == 0:
                break
        if num_splits == 0:
            break
    print(all_splits)

    for split in all_splits:
        print(f"Split -> {split}")
        left = np.array([])
        right = np.array([])
        for i, a in enumerate(x):
            if a in split:
                left = np.append(left, y[i])
            else:
                right = np.append(right, y[i])

        impr_red = get_impr_red(left, right, max_impr)
        print(f"Cur. Impurity Reduction = {impr_red}\n")

        if impr_red > best_red:
            best_red = impr_red
            best_split = split
        print(f"Best split {best_split} has {best_red} impurity reduction\n")

    return best_split
