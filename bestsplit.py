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
        impr_red = 0.25 - (freq_lower*gini_lower) - ((1-freq_lower)*gini_higher)

        print(f"Cur. Split Impurity Reduction = {impr_red}\n")
        if impr_red > best_red:
            best_red = impr_red
            best_split = c

        print(f"Best split {best_split} has {best_red} impurity reduction\n")
    return best_split
