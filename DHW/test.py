import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

import numpy as np
import math
from scipy.special import comb
from sklearn import cluster
from sklearn import neighbors
import copy
from tqdm import tqdm


def test(filename, test_ft, test_labels):

    metrics_path = os.path.join('metrics')
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path, exist_ok=True)
    metrics = {}

    radius = [1, 2, 4, 8]
    r_res = recall_r(test_ft, test_labels, radius)
    for i in range(len(radius)):
        print('R@{}: {}'.format(radius[i], r_res[i]))
        metrics['R@{}'.format(radius[i])] = r_res[i]



    unique_labels = np.unique(test_labels)
    n_clusters = unique_labels.shape[0]
    nmi_res, f1_res = evaluate_cluster(test_ft, test_labels, n_clusters)

    print('NMI: {}'.format(nmi_res))
    metrics['NMI'] = nmi_res
    print('F1: {}'.format(f1_res))
    metrics['F1'] = f1_res

    return metrics



def dist_func(a, b):
    dist = -2 * np.matmul(a, np.transpose(b)) + np.sum(np.square(a), 1).reshape(1, -1) + np.sum(
        np.square(b), 1).reshape(-1, 1)
    return dist


def recall_r(ft, labels, radius):


    dist = dist_func(ft, ft)
    max_radius = max(radius)
    indices = dist.argsort()[:, 1:max_radius+1]
    hit = labels[indices] == np.expand_dims(labels, 1)
    res = np.zeros_like(radius, dtype=np.float)
    for i in range(len(radius)):
        res[i] = np.mean(np.sum(hit[:, :radius[i]], 1).astype(bool))
    return res


def nmi_f1(ft, labels):
    n_samples = len(ft)
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.shape[0]
    k_means = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(ft)
    # NMI
    nmi_res = normalized_mutual_info_score(labels, k_means.labels_, average_method='geometric')
    # F1 score
    unique_centers = np.unique(k_means.labels_)
    matrix = np.zeros([len(unique_labels), len(unique_centers)])
    weight = np.zeros_like(unique_labels, dtype=float)
    for i in range(n_clusters):
        label = unique_labels[i]
        weight[i] = np.sum(labels == label).astype(float) / n_samples
        for j in range(n_clusters):
            center = unique_centers[j]
            label_ind = labels == label
            center_ind = k_means.labels_ == center
            inter = np.sum(label_ind*center_ind).astype(float)
            prec = inter / np.sum(center_ind, dtype=float)
            recall = inter / np.sum(label_ind, dtype=float)
            if prec + recall == 0:
                continue
            f = 2*prec*recall/(prec+recall)
            matrix[i, j] = f
    f1_res = np.max(matrix, 1)
    f1_res = np.sum(f1_res * weight)

    return nmi_res, f1_res




def evaluate_cluster(feats, labels, n_clusters):

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=1).fit(feats)
    centers = kmeans.cluster_centers_

    # k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    idx_in_centers = neigh.predict(feats)
    num = len(feats)
    d = np.zeros(num)
    for i in range(num):
        d[i] = np.linalg.norm(feats[i, :] - centers[idx_in_centers[i], :])

    labels_pred = np.zeros(num)
    for i in np.unique(idx_in_centers):
        index = np.where(idx_in_centers == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid
    nmi, f1 = compute_clutering_metric(labels, labels_pred)
    return nmi, f1

def compute_clutering_metric(idx, item_ids):

    N = len(idx)


    centers = np.unique(idx)
    num_cluster = len(centers)

    count_cluster = np.zeros(num_cluster)
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0])


    keys = np.unique(item_ids)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])


    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[item_ids[i]]
        count_item[index] = count_item[index] + 1


    purity = 0
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)


    count_cross = np.zeros((num_cluster, num_item))
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0]
        index_item = item_map[item_ids[i]]
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1


    I = 0
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s


    H_cluster = 0
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)


    tp_fp = 0
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)


    tp = 0
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)


    fp = tp_fp - tp


    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp


    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return NMI, F




