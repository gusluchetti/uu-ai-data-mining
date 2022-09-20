import math
import numpy as np
import pandas as pd
from gini import gini_impurity


def load_dataset(path):
    df = pd.read_csv(path)
    print(f"Loaded dataframe.\n {df.head()}")
    return df


def tree_grow(x, y, nmin: int, minleaf: int, nfeat):
    return "tree"


def tree_pred():
    return "tree"


x = [[1], [2]]
y = [0, 1]

load_dataset("data.csv")
