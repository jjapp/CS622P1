import numpy as np
import test_utility as tu
import scipy.spatial as sp


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    num_rows = np.shape(X_train)[0]
    prediction = []
    for row in X_test:
        vote_list=[]
        for j in range(num_rows):
            distance = sp.distance.euclidean(row, X_train[j])
            vote_list.append((distance, Y_train[j]))

        # sort the list of tuples using a bubble sort
        for i in range(len(vote_list)):
            for j in range(0, len(vote_list)-i-1):
                if vote_list[j][0] > vote_list[j+1][0]:
                    temp = vote_list[j]
                    vote_list[j] = vote_list[j + 1]
                    vote_list[j + 1] = temp

        # slice the list at K
        vote_list = vote_list[:K]

        # get the vote
        vote = 0
        for row in vote_list:
            vote = vote + row[1]

        if vote <0:
            vote = -1
        else:
            vote = 1
        prediction.append(vote)

    # get accuracy
    true_votes = []
    false_votes = []
    for i in range(len(prediction)):
        if i == Y_test[i]:
            true_votes.append(1)
        else:
            false_votes.append(1)
    accuracy = sum(true_votes)/(sum(true_votes)+sum(false_votes))
    return accuracy


def choose_K(X_train, Y_train, X_val, Y_val):

    max_k = np.shape(X_train)[0]
    best_accuracy = 0
    best_k = 0
    for i in range(1, max_k):
        test_acc =KNN_test(X_train, Y_train, X_val, Y_val, i)
        if test_acc > best_accuracy:
            best_k = i
            best_accuracy = test_acc
    return best_k

if __name__=="__main__":
    x, y = tu.load_data('data_4.txt')

    # split the data
    rows = np.shape(x)[0]

    X_train = x[:3]
    Y_train = y[:3]

    X_test = x[3:]
    Y_test = y[3:]

    f = choose_K(X_train, Y_train, X_test, Y_test)
    print (f)