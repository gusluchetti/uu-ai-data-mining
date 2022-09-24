import numpy as np
from math import pow
from gini import gini_impurity, get_max_impurity
from itertools import combinations


# TODO: throw repeated code lines into separate function
# (left and right and beyond)


# bestsplit for numerical values
def get_bestsplit(x, y):
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
        gini_left = gini_impurity(left)
        gini_right = gini_impurity(right)
        freq_l = len(left)/len(x)
        freq_r = 1 - freq_l
        print(gini_left, gini_right, freq_l, freq_r)

        impr_red = get_max_impurity(np.unique(y)) - ((freq_l*gini_left) + ((freq_r)*gini_right))
        print(f"Cur. Impurity Reduction = {impr_red}\n")
        if impr_red > best_red:
            best_red = impr_red
            best_split = split
        print(f"Best split {best_split} has {best_red} impurity reduction\n")

    return best_split


# bestsplit for categorical values
def get_bestsplit_cat(x, y):
    best_red = float('-inf')
    best_split = None
    x_uniques = np.unique(x)
    length = x_uniques.size
    # 2 to the power of L-1 minus 1 where L is unique categorical variables
    num_splits = pow(2, length - 1) - 1
    # should finish with a list of all splits i.e. [a, b, c, d ,ab, ac, ad]
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
        gini_left = gini_impurity(left)
        gini_right = gini_impurity(right)
        freq_l = len(left)/len(x)
        freq_r = 1 - freq_l
        print(gini_left, gini_right, freq_l, freq_r)

        impr_red = get_max_impurity(np.unique(y)) - ((freq_l*gini_left) + ((freq_r)*gini_right))
        print(f"Cur. Impurity Reduction = {impr_red}\n")
        if impr_red > best_red:
            best_red = impr_red
            best_split = split
        print(f"Best split {best_split} has {best_red} impurity reduction\n")

    return best_split
