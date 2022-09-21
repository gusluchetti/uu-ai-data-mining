import numpy as np
from gini import gini_impurity


def bestsplit(x, y):
    best_red = float('-inf')
    best_split = None

    x_sorted = np.sort(np.unique(x))
    x_splitpoints = (
       x_sorted[0:(len(x)-3)] + x_sorted[1:(len(x)-2)]
    )/2
    print(x, y, x_splitpoints, "\n")
    # for each splitpoint, calculate "if lower" split impurity
    for i, c in enumerate(x_splitpoints):
        print(f"Current Split = {c}")
        lower = y[x < c]
        gini_lower = gini_impurity(lower)
        higher = y[x >= c]
        gini_higher = gini_impurity(higher)

        freq_lower = len(lower)/len(x)
        freq_higher = len(higher)/len(x)
        impr_red = 1/4 - (freq_lower*gini_lower) - (freq_higher*gini_higher)

        print(f"Cur. Split Impurity Reduction = {impr_red}\n")
        if impr_red > best_red:
            best_red = impr_red
            best_split = c
        print(f"Best split {best_split} had {impr_red} impurity reduction\n")

    return best_split
