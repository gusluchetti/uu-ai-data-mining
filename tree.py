import numpy as np
import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)
    print(f"Loaded dataframe.\n {df.info()}\n")
    return df


def tree_grow(x, y, nmin: int, minleaf: int, nfeat):
    return "tree"


def tree_pred():
    return "pred"


df = load_dataset("data.csv")
x = df.loc[:-2]
y = df["class"]
print(x, y, "\n")

for column in df:
    print(f"Col: {column}\n{df[column]}\n")

x = np.array([[]])
y = np.array([])
