import tree
import numpy as np
import time


# testing our code using the Pima indians data
def test_pima():
    pima_data = np.genfromtxt('./data/pima.txt', delimiter=',')
    x = pima_data[:, :8]
    y = pima_data[:, 8]
    t = tree.tree_grow(x, y, 20, 5, 8)
    yhat = tree.tree_pred(x, t)

    # calculate accuracy, precision, recall
    pred0_tru0 = 0
    pred0_tru1 = 0
    pred1_tru0 = 0
    pred1_tru1 = 0

    for i in range(len(y)):
        if int(yhat[i]) == 0 and int(y[i]) == 0:
            pred0_tru0 += 1
        elif int(yhat[i]) == 0 and int(y[i]) == 1:
            pred0_tru1 += 1
        elif int(yhat[i]) == 1 and int(y[i]) == 0:
            pred1_tru0 += 1
        else:
            pred1_tru1 += 1

    print(f"These are the 0-0 {pred0_tru0} / 0-1 {pred0_tru1} / 1-0 {pred1_tru0} / 1-1 {pred1_tru1}")
    print(f"Precision class 0 {pred0_tru0 / (pred0_tru0 + pred0_tru1)}")
    print(f"Recall class 0 {pred0_tru0 / (pred0_tru0 + pred1_tru0)}")
    print(f"Precision class 1 {pred1_tru1 / (pred1_tru1 + pred1_tru0)}")
    print(f"Recall class 1 {pred1_tru1 / (pred1_tru1 + pred0_tru1)}")
    print(f"Accuracy {(pred1_tru1 + pred0_tru0) / len(y)}")


def test_eclipse_1():
    # TRAINING
    eclipse_training_data = np.genfromtxt('./data/eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)

    # prepare the x matrix
    tr_col_pre = eclipse_training_data[:, 2]  # column pre-release bugs
    tr_col_pre = tr_col_pre.reshape(len(eclipse_training_data), 1)
    tr_col_rest = eclipse_training_data[:, 4:44]  # other columns
    x = np.hstack((tr_col_pre, tr_col_rest))  # binding the pre-release column with other columns

    # prepare the y column
    y = eclipse_training_data[:, 3]  # the target column
    y = [1 if y[i] > 0 else 0 for i in range(len(y))]

    # grow a single tree
    t1 = tree.tree_grow(x, y, 15, 5, 41)

    '''
    # Uncomment this section to check the split points & classes of the first 3 leaf nodes
    print(f"PARENT split col {t1.split_col} split point {t1.split_point}")
    print("Number of children:", len(t1.children))
    print("LEFT-1 split col", t1.children[0].split_col, "split point", t1.children[0].split_point)
    print("RIGHT-1 split col", t1.children[1].split_col, "split point", t1.children[1].split_point)

    t1_left = t1.children[0]
    print("LEFT-2 split col", t1_left.children[0].split_col, "split point", t1_left.children[0].split_point)
    print("RIGHT-2 split col", t1_left.children[1].split_col, "split point", t1_left.children[1].split_point)
    print("LEFT-2 leaf class ==", tree.find_leaf_class(t1_left.children[0].matrix))
    print("RIGHT-2 left class ==", tree.find_leaf_class(t1_left.children[1].matrix))
    print("RIGHT-1 left class ==", tree.find_leaf_class(t1.children[1].matrix))
    '''

    # TESTING
    eclipse_testing_data = np.genfromtxt('./data/eclipse-metrics-packages-3.0.csv', delimiter=';', skip_header=True)

    # pre-processing the data
    ts_col_pre = eclipse_testing_data[:, 2]
    ts_col_pre = ts_col_pre.reshape(len(eclipse_testing_data), 1)
    ts_col_rest = eclipse_testing_data[:, 4:]
    x_test = np.hstack((ts_col_pre, ts_col_rest))

    y_test = eclipse_testing_data[:, 3]
    y_test = [1 if y_test[i] > 0 else 0 for i in range(len(y_test))]

    # make prediction
    pred1 = tree.tree_pred(x_test, t1)

    # calculate accuracy, precision, recall
    pred0_tru0 = 0
    pred0_tru1 = 0
    pred1_tru0 = 0
    pred1_tru1 = 0

    for i in range(len(y_test)):
        if int(pred1[i]) == 0 and int(y_test[i]) == 0:
            pred0_tru0 += 1
        elif int(pred1[i]) == 0 and int(y_test[i]) == 1:
            pred0_tru1 += 1
        elif int(pred1[i]) == 1 and int(y_test[i]) == 0:
            pred1_tru0 += 1
        else:
            pred1_tru1 += 1

    print(f"Predicted 0 - Truth 0 = {pred0_tru0} \nPredicted 0 - Truth 1 = {pred0_tru1}")
    print(f"Predicted 1 - Truth 0 = {pred1_tru0} \nPredicted 1 - Truth 1 = {pred1_tru1}")
    print(f"Precision class 0 = { pred0_tru0 / (pred0_tru0 + pred0_tru1) }")
    print(f"Recall class 0 = { pred0_tru0 / (pred0_tru0 + pred1_tru0) }")
    print(f"Precision class 1 = { pred1_tru1 / (pred1_tru1 + pred1_tru0) }")
    print(f"Recall class 1 = { pred1_tru1 / (pred1_tru1 + pred0_tru1) }")
    print(f"Accuracy = { (pred1_tru1 + pred0_tru0) / len(y_test) }")

def test_eclipse_2(nmin, minleaf, nfeat, m):
    # TRAINING
    eclipse_training_data = np.genfromtxt('./data/eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)

    # prepare the x matrix
    tr_col_pre = eclipse_training_data[:, 2]  # column pre-release bugs
    tr_col_pre = tr_col_pre.reshape(len(eclipse_training_data), 1)
    tr_col_rest = eclipse_training_data[:, 4:]  # other columns
    x = np.hstack((tr_col_pre, tr_col_rest))  # binding the pre-release column with other columns

    # prepare the y column
    y = eclipse_training_data[:, 3]  # the target column
    y = [1 if y[i] > 0 else 0 for i in range(len(y))]

    # grow the trees
    forest = tree.tree_grow_b(x, y, nmin, minleaf, nfeat, m)

    # TESTING
    eclipse_testing_data = np.genfromtxt('./data/eclipse-metrics-packages-3.0.csv', delimiter=';', skip_header=True)

    # pre-processing the data
    ts_col_pre = eclipse_testing_data[:, 2]
    ts_col_pre = ts_col_pre.reshape(len(eclipse_testing_data), 1)
    ts_col_rest = eclipse_testing_data[:, 4:]
    x_test = np.hstack((ts_col_pre, ts_col_rest))

    y_test = eclipse_testing_data[:, 3]
    y_test = [1 if y_test[i] > 0 else 0 for i in range(len(y_test))]

    # make prediction
    pred1 = tree.tree_pred_b(x_test, forest)

    # calculate accuracy, precision, recall
    pred0_tru0 = 0
    pred0_tru1 = 0
    pred1_tru0 = 0
    pred1_tru1 = 0

    for i in range(len(y_test)):
        if int(pred1[i]) == 0 and int(y_test[i]) == 0:
            pred0_tru0 += 1
        elif int(pred1[i]) == 0 and int(y_test[i]) == 1:
            pred0_tru1 += 1
        elif int(pred1[i]) == 1 and int(y_test[i]) == 0:
            pred1_tru0 += 1
        else:
            pred1_tru1 += 1

    print(f"Predicted 0 - Truth 0 = {pred0_tru0} \nPredicted 0 - Truth 1 = {pred0_tru1}")
    print(f"Predicted 1 - Truth 0 = {pred1_tru0} \nPredicted 1 - Truth 1 = {pred1_tru1}")
    print(f"Precision class 0 = { pred0_tru0 / (pred0_tru0 + pred0_tru1) }")
    print(f"Recall class 0 = { pred0_tru0 / (pred0_tru0 + pred1_tru0) }")
    print(f"Precision class 1 = { pred1_tru1 / (pred1_tru1 + pred1_tru0) }")
    print(f"Recall class 1 = { pred1_tru1 / (pred1_tru1 + pred0_tru1) }")
    print(f"Accuracy = { (pred1_tru1 + pred0_tru0) / len(y_test) }")

# test_pima()
#test_eclipse_1()
print("calculating...")
start_time_total = time.perf_counter()

test_eclipse_1()

#test_eclipse_2(nmin = 15, minleaf = 5, nfeat = 41, m = 100)

#test_eclipse_2(nmin = 15, minleaf = 5, nfeat = 6, m = 100)

end_time_total = time.perf_counter()
print("total time:", end_time_total - start_time_total)