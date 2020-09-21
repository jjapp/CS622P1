import numpy as np
import test_utility as tu
import random
import scipy.spatial as sp


def assign_to_cluster(cluster_dict, X, mu):
    # assign each point to a cluster
    for i in X:
        # get distances to all mu
        distance_list = []
        for j in mu:
            distance = sp.distance.euclidean(i, j)
            distance_list.append(distance)
        min_index = distance_list.index(min(distance_list))
        cluster_dict[min_index].append(i)
    return cluster_dict


def get_centers(cluster_dict):
    new_mu_list = []
    for key in cluster_dict:
        new_mu = np.mean(cluster_dict[key], axis=0)
        new_mu_list.append(new_mu)
    new_mu_array = np.vstack(new_mu_list)
    return new_mu_array


def get_max_distance(ctr1, ctr2):
    dist_list=[]
    for i in range(len(ctr1)):
        d = sp.distance.euclidean(ctr1[i], ctr2[i])
        dist_list.append(d)
    return max(dist_list)


def K_Means(X, K, mu):
    if len(mu) == 0:
        max_value = np.amax(X)
        mu = []
        dimension = np.shape(X)[1]
        for i in range(K):
            coord_list = []
            for j in range(dimension):
                coord_list.append(random.randint(0, max_value))
            point = np.array(coord_list)
            mu.append(point)

    # create a dictionary to hold cluster points
    cluster_dict = {}
    for idx in range(len(mu)):
        cluster_dict[idx] = []

    new_dict = assign_to_cluster(cluster_dict, X, mu)
    new_centers = get_centers(new_dict)
    return new_centers


def K_Means_better(X, K):
    centers = K_Means(X, K)
    new_centers = K_Means(X, K, centers)
    d = get_max_distance(centers, new_centers)
    while d != 0:
        centers = new_centers
        new_centers = K_Means(X, K, centers)
        d = get_max_distance(centers, new_centers)
    return centers

if __name__ == '__main__':
    x, y = tu.load_data('data_6.txt')
    model = K_Means_better(x, 2)

