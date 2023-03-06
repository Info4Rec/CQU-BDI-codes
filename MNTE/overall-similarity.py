import scipy.io as sio
import numpy as np

def i2t(sims):

    npts = len(sims)
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index][:npts * 5])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    rsum = r1 + r5 + r10
    return (r1, r5, r10, rsum)

def t2i(sims):
    npts = len(sims)
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    rsum = r1 + r5 + r10
    return (r1, r5, r10, rsum)


def print_float(r):
    print("%.1f, %.1f, %.1f, %.1f" % r)


def calc_g_l_r_d(path_global, path_l_r, path_d, rate_l_r=1, rate_d=1, rate_g=2, nums=1000):
    """
    Calculate the overall similarity
    """
    similarity_global = sio.loadmat(path_global)['similarity']

    similarity_l_r = sio.loadmat(path_l_r)['similarity']
    similarity_st = sio.loadmat(path_d)['similarity']
    similarity_overall = similarity_l_r * rate_l_r + similarity_st * rate_d + similarity_global * rate_g
    similarity_overall_part = similarity_overall[:nums]

    r = i2t(similarity_overall_part)
    print("Image to Text: ", end='')
    print_float(r)
    r1 = t2i(similarity_overall_part)
    print("Text to Image: ", end='')
    print_float(r1)


if __name__ == '__main__':
    # path_global: the path of sims_global.mat
    path_g = './global/sims/sims_global.mat'
    # path_relation: the path of sims_relation.mat
    path_l_r = './relation/sims/sims_relation.mat'
    # path_digital: the path of sims_digital.mat
    path_d = './digital/sims/sims_digital.mat'

    # rate of subnetwork
    rate_l_r = 1
    rate_d = 2
    rate_g = 2

    calc_g_l_r_d(path_global=path_g, path_l_r=path_l_r, path_d=path_d, rate_l_r=rate_l_r, rate_d=rate_d, rate_g=rate_g)

