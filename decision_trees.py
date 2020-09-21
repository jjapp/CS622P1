import numpy as np
import math
import test_utility as tu


def get_entropy(y):
    v1 = 0
    v2 = 0
    for i in y:
        if i == 0:
            v1 += 1
        elif i == 1:
            v2 += 1
        else:
            print('Error: Unexpected Value for Label')
    tot_obs = v1 + v2

    if v1 == 0:
        probv1 = 0
    else:
        probv1 = v1 / tot_obs

    if v2 == 0:
        probv2 = 0
    else:
        probv2 = v2 / tot_obs

    try:
        entropy = -(probv1) * math.log(probv1, 2) - (probv2) * math.log(probv2, 2)
    except:
        entropy = 0

    return entropy


def get_max_gain_idx(z):
    gain_list = []
    width = np.shape(z)[1]
    rows = np.shape(z)[0]
    h0 = get_entropy(z[:, -1])
    for i in range((width - 1)):
        arr1 = z[z[:, i] == 0]
        arr2 = z[z[:, i] == 1]

        # get number of rows in arr1
        rows1 = np.shape(arr1)[0]
        rows2 = np.shape(arr2)[0]

        # get h for each row
        h1 = get_entropy(arr1[:, -1])
        h2 = get_entropy(arr2[:, -1])
        gain = h0 - ((rows1 / rows) * h1 + (rows2 / rows) * h2)
        gain_list.append(gain)
    max_idx = gain_list.index(max(gain_list))
    max_gain = max(gain_list)
    partition = Partition(max_idx, z[0, max_idx:])

    return partition, max_gain


def split_array(z, idx):
    yes_rows = z[z[:, idx] == 0]
    no_rows = z[z[:, idx] == 1]

    return yes_rows, no_rows

'''
def get_vote(z, idx):
    yes = z[z[:, idx] == 0]
    no = z[z[:, idx] == 1]

    yes_rows = np.shape(yes)[0]
    no_rows = np.shape(no)[0]

    yes_sum = np.sum(yes[:, -1])
    no_sum = np.sum(no[:, -1])

    if yes_sum / yes_rows < 0.5:
        yes_vote = 0
    else:
        yes_vote = 1
    if no_sum / no_rows < 0.5:
        no_vote = 0
    else:
        no_vote = 1

    return yes_vote, no_vote

'''
class Partition:
    """Stores the index position for a question and
    allows you to compare value to a feature value"""

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

    def compare(self, example):
        val = example[self.idx]
        return val == self.value[self.idx]

    def __repr__(self):
        return "Index " + str(self.idx)


class Leaf:
    """Holds the classification for a leaf in the tree"""

    def __init__(self, z):
        total_rows = np.shape(z)[0]
        total_no = np.sum(z[:, -1])
        total_yes = total_rows - total_no
        if total_no - total_yes > 0:
            self.prediction = 1
        else:
            self.prediction = 0


class DecisionNode:
    """ Stores the decision point and child nodes

    I define 0 == yes, 1 == no"""

    def __init__(self, partition, yes_branch, no_branch):
        self.partition = partition
        self.yes_branch = yes_branch
        self.no_branch = no_branch


def build_tree(z, max_depth):

    # get first index
    gain = get_max_gain_idx(z)

    # check to see if we're at zero gain
    if gain[1] == 0:
        return Leaf(z)

    if max_depth == 0:
        return Leaf(z)

    max_depth = max_depth - 1

    # if gain is not zero we need to split and continue
    yes_rows, no_rows = split_array(z, gain[0].idx)

    # build the yes branch
    yes_branch = build_tree(yes_rows, max_depth)

    # build the no branch
    no_branch = build_tree(no_rows, max_depth)

    return DecisionNode(gain[0], yes_branch, no_branch)


def DT_train_binary(X, Y, max_depth):
    z = np.column_stack((X, Y))  # merge the two arrays
    model = build_tree(z, max_depth)
    return model


def DT_make_prediction(x, DT):
    """ This function should take a single sample and a
    trained decision tree and return a single classification.
    The output should be a scalar value."""

    # Check if we are at a leaf
    if isinstance(DT, Leaf):
        return DT.prediction

    # follow yes or no branch
    if DT.partition.compare(x):
        return DT_make_prediction(x, DT.yes_branch)

    else:
        return DT_make_prediction(x, DT.no_branch)


def DT_test_binary(X, Y, DT):
    pred_list = []
    true_pred = []
    false_pred = []
    for i in X:
        pred = DT_make_prediction(i, DT)
        pred_list.append(pred)

    for j in range(len(pred_list)):
        if pred_list[j] == Y[j]:
            true_pred.append(1)
        else:
            false_pred.append(1)

    true = sum(true_pred)
    false = sum(false_pred)

    accuracy = true / (true + false)

    return accuracy


def print_tree(node):
    if isinstance(node, Leaf):
        print(node.prediction)
        return

    print(node.partition)

    print_tree(node.yes_branch)
    print_tree(node.no_branch)


if __name__ == "__main__":
    x, y = tu.load_data('data_2.txt')
    H = DT_train_binary(x, y, 2)
    pred = DT_test_binary(x, y, H)
    print (pred)
