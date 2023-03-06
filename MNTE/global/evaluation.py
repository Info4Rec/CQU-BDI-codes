from __future__ import print_function

from data import get_test_loader
import time
import tqdm
import numpy as np
import torch
from collections import OrderedDict
from utils import dot_sim, cosine_sim, get_model
import logging
import scipy.io as sio
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lengths = []

    for i, (images, targets, cap_length, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_length = model.forward_emb(images, targets, cap_length)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = torch.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = torch.zeros((len(data_loader.dataset), cap_emb.size(1)))
            img_embs[ids, :] = img_emb.cpu()
            cap_embs[ids, :] = cap_emb.cpu()
            cap_lengths.extend(cap_length)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Encoding Test data: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del images, targets

    return img_embs, cap_embs, cap_lengths


def evalrank(opt, checkpoint, split='dev', fold5=False):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # construct model
    model = get_model(opt)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_test_loader(opt, workers=4, split_name=split)

    print('Encoding data...')

    img_embs, cap_embs, cap_lenghts = encode_data(
        model, data_loader, opt.log_step, logging.info)
    torch.cuda.empty_cache()


    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    img_embs = img_embs.cpu().numpy()
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    cap_embs = cap_embs.cpu().numpy()

    measure = opt.measure
    if measure == 'cosine':
        sim_matrix_fn = cosine_sim
    elif measure == 'dot':
        sim_matrix_fn = dot_sim

    sims = shard_xattn(sim_matrix_fn, img_embs, cap_embs)

    # image to text
    (r1, r5, r10), rsum = i2t(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, rsum))

    # text to image
    (r1i, r5i, r10i), risum = t2i(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, risum))

    # Save the similarity matrix
    print('Saving results...')
    dir = './sims/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    sio.savemat('%s/sims_global.mat' % dir, {'similarity': sims})
    print('Saving success...')


def shard_xattn(sim_matrix_fn, img_embs, cap_embs, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    d = np.zeros((len(img_embs), len(cap_embs)))
    for i in tqdm.trange(n_im_shard):
        im_start, im_end = shard_size * \
                           i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * \
                                 j, min(shard_size * (j + 1), len(cap_embs))
            im = img_embs[im_start:im_end]
            s = cap_embs[cap_start:cap_end]
            sim = sim_matrix_fn(im, s)
            d[im_start:im_end, cap_start:cap_end] = sim
    return d


def i2t(sims):
    npts = len(sims)
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    # sims = sims.T

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
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    rsum = r1 + r5 + r10
    return (r1, r5, r10), rsum


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
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    rsum = r1 + r5 + r10
    return (r1, r5, r10), rsum
