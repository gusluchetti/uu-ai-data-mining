import numpy as np
import random


class Node:
    def __init__(self, matrix):
        self.children = []
        self.matrix = matrix
        self.split_col = None
        self.split_point = None
        self.leaf_class = None


def tree_grow(x, y, nmin, minleaf, nfeat) -> Node:
    """Creating a Tree model based on the training data as input

    Parameters
        list x: 2-dimensional array of numerical values
        list y: 1-dimensional array of class labels (binary, 0 or 1)
        int nmin: minimum number of observations in a node to be split
        int minleaf: minimum number of observations in a leaf-node
        int nfeat: number of features to be considered for each split
    Returns
        (Tree) the Classification Tree object
    """
    _y = np.array(y).reshape(len(x), 1)
    matrix = np.hstack((x, _y))
    root = Node(matrix)

    split(root, nmin, minleaf, nfeat)
    return root


def tree_pred(x, tr) -> list:
    """Predicting the classes of every observations in the input x

    Parameters
        x (list): 2-dimensional array of numerical values
        tr (Tree): the Tree model
    Returns:
        (list) A list of class labels
    """
    predictions = []
    for i in x:
        c = predict(i, tr)
        predictions.append(c)
    return predictions


def tree_grow_b(x, y, nmin, minleaf, nfeat, m) -> list:
    _y = np.array(y).reshape(len(x), 1)
    matrix = np.hstack((x, _y))
    samples = create_bootstrap_samples(matrix, m)
    roots = []

    for s in samples:
        root = Node(s)
        split(root, nmin, minleaf, nfeat)
        roots.append(root)

    return roots


def tree_pred_b(x, trees):
    predictions = []

    for i in x:
        votes = []
        for t in trees:
            votes.append(predict(i, t))

        # find the most voted
        vals, occs = np.unique(votes, return_counts=True)
        dic = {o:v for o, v in zip(occs, vals)}
        predictions.append(dic[occs.max()])

    return predictions


def split(node, nmin, minleaf, nfeat):
    if len(node.matrix) < nmin or len(node.matrix) < minleaf:
        # leaf node is reached
        node.leaf_class = find_leaf_class(node.matrix)
    else:
        # attempt splitting
        split_col, split_pt, split_left, split_right = find_best_split(node.matrix, nfeat, minleaf)

        if split_col is not None:
            # a valid split is found
            left_node = Node(split_left)
            right_node = Node(split_right)

            node.children.extend([left_node, right_node])
            node.split_col = split_col
            node.split_point = split_pt

            split(left_node, nmin, minleaf, nfeat)
            split(right_node, nmin, minleaf, nfeat)
        else:
            # leaf node is reached
            node.leaf_class = find_leaf_class(node.matrix)


def predict(instance, node):
    if node.leaf_class is not None:
        return int(node.leaf_class)
    elif instance[node.split_col] <= node.split_point:
        return predict(instance, node.children[0])
    else:
        return predict(instance, node.children[1])


def traverse(node):
    print(f"============== NODE")
    print(f"Split column {node.split_col}")
    print(f"Split point {node.split_point}")
    print(f"Leaf class {node.leaf_class}")

    if len(node.children) > 0:
        for i in range(len(node.children)):
            traverse(node.children[i])
    else:
        return


def create_bootstrap_samples(matrix, m):
    samples = []
    for i in range(m):
        sample = []
        for j in range(len(matrix)):
            rand_row = random.randint(0, len(matrix) - 1)
            sample.append(matrix[rand_row])
        samples.append(sample)
    return samples


def find_best_split(matrix, nfeat, minleaf=1):
    """Iterates over the attributes and instances in the training data. Each time performs a binary split
    and calculates the impurity reduction.

    Parameters
        matrix: (list) 2-dimensional numerical array. The last column is the class labels (Y), other columns are the attributes (X).
        minlength: (int) The minimum size of a split.
    Returns
        The best left and right split. Both are 2-dimensional subset arrays of the input data.
        Or None if no binary split that fits the criteria is found.
    """
    unsplit = np.array(matrix)
    attr_cols = []

    best_reduction = 0
    best_column = None
    best_split_point = None
    best_split_left = None
    best_split_right = None

    # designate attribute columns
    if nfeat < (len(unsplit[0]) - 1):
        attr_cols = random.sample(range(len(unsplit[0]) - 1), nfeat)
    else:
        attr_cols = list(range(len(unsplit[0]) - 1))

    for col in attr_cols:  # Iterate over the attribute columns
        split_points = find_split_points(unsplit[:, col])

        for point in split_points:  # Iterate over the possible splits in a column
            split_left, split_right = split_matrix(unsplit, col, point)

            if len(split_left) >= minleaf and len(split_right) >= minleaf:
                # this split meets the minleaf constraint
                lastidx = len(unsplit[0]) - 1
                y = unsplit[:, lastidx]
                y_left = split_left[:, lastidx]
                y_right = split_right[:, lastidx]
                reduction = calculate_impurity_reduction(y, y_left, y_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_column = col
                    best_split_point = point
                    best_split_left = split_left
                    best_split_right = split_right

    return best_column, best_split_point, best_split_left, best_split_right


def split_matrix(matrix, column, split_point):
    mat = np.array(matrix)
    left = mat[np.where(mat[:, column] <= split_point)]
    right = mat[np.where(mat[:, column] > split_point)]
    return left, right


def find_split_points(arr: list) -> list:
    unique_vals = np.sort(np.unique(arr))
    length = len(unique_vals)
    split_points = (unique_vals[0:(length-1)] + unique_vals[1:length]) / 2
    return split_points


def sort_by_column(arr, col_idx: int):
    """Sorts a 2D numerical array by a column.
    :param arr: 2-dimensional array of numbers.
    :param col_idx: index of the sorting column, starts from 0.
    :return: sorted 2-dimensional array.
    """
    sorted_idx = arr[:, col_idx].argsort()
    sorted_arr = arr[sorted_idx]
    return sorted_arr


def find_leaf_class(matrix):
    mat = np.array(matrix)
    vals, occurrences = np.unique(mat[:, len(mat[0])-1], return_counts=True)
    # dic = dict(zip(occurrences, vals))
    dic = { o: v for o, v in zip(occurrences, vals) }
    return dic[occurrences.max()]


def calculate_impurity_reduction(y_parent, y_left, y_right) -> float:
    """Parent's Gini impurity - children's Gini impurity

    Parameters
        y_parent: (1-d numeric array) the labels of the parent node
        y_left: (1-d numeric array) left subset of y_parent
        y_right: (1-d numeric array) right subset of y_parent
    Returns
        (float) Amount of impurity reduction
    """
    im_parent = calculate_gini_impurity(y_parent)
    im_left = calculate_gini_impurity(y_left)
    im_right = calculate_gini_impurity(y_right)
    w_left = len(y_left) / len(y_parent)
    w_right = len(y_right) / len(y_parent)
    return im_parent - (w_left * im_left) - (w_right * im_right)


def calculate_gini_impurity(arr: list) -> float:
    """Calculates the Gini impurity of a single node.

    :param arr: a list of class labels (1-dimensional array). It is assumed that the labels are only 0 and 1.
    :return: gini impurity number.
    """
    unique_vals, count_vals = np.unique(arr, return_counts=True)
    if len(unique_vals) == 1:
        return 0
    else:
        gini = 1
        for i in range(len(unique_vals)):
            gini = gini * (count_vals[i] / len(arr))
        return giniimport numpy as np
