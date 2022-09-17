import numpy as np
from math import pow

def gini_impurity(array):
    # 1 - p1squared - p2squared

    length = len(array)
    print(f'Vector length -> {length}')
    count_0 = np.count_nonzero(array == 0)
    print(f'Number of 0s -> {count_0}')
    count_1 = np.count_nonzero(array == 1)
    print(f'Number of 1s -> {count_1}')
    p1squared = pow(count_0/length, 2)
    p2squared = pow(count_1/length, 2)

    return 1 - p1squared - p2squared

array = np.array([1,0,1,1,1,0,0,1,1,1,0])
gini = gini_impurity(array)

print(f'Expected gini: {0.23140495867768596}')
print(f'Calculated gini: {gini}')