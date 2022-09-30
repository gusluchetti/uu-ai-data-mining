import numpy as np
import math


# max possible impurity is related to the number of unique classes/values
def get_max_impurity(array):
    """array of unique elements"""
    length = len(array)
    return 1-length*(math.pow(1/length, 2))


def gini_impurity(array):
    # 1 - sum (p)^2
    """ updated version of gini impurity to handle separation
    between a number N of different classes
    classe could either be numbers or chars

    args: np array of N classes
    returns: weighted gini_index of node
    """
    print(f"Node being evaluated -> {array}")
    length = len(array)
    uniques = np.unique(array)
    print(f"Classes -> {uniques}")
    freqs = np.array([])
    for label in uniques:
        count = np.where(array == label)[0].size
        freq_p = count/length
        freqs = np.append(freqs, [freq_p])
    print(f"Freq. for each class -> {freqs}")
    sum = 0
    for freq in freqs:
        sum += math.pow(freq, 2)
    gini = 1 - sum
    if freqs[0] == 1:  # for single element nodes
        gini = 0.0
    print(f"Gini = {gini}\n")
    return gini
