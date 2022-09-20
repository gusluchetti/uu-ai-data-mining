import math
import numpy as np
import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)
    print(f"Loaded dataframe.\n {df.describe()}")
    return df


# gini index for two classes (0, 1)
def gini_impurity(array):
    length = len(array)
    count_0 = np.count_nonzero(array == 0)
    count_1 = np.count_nonzero(array == 1)

    p1squared = math.pow(count_0/length, 2)
    p2squared = math.pow(count_1/length, 2)
    return (1 - p1squared - p2squared)


def tree_grow(x, y, nmin: int, minleaf: int, nfeat):
    return "tree"


def tree_pred():
    return "tree"


x = [[1], [2]]
y = [0, 1]

load_dataset("data.csv")
