# encoding:utf-8

import os, random, copy
import torch
import argparse
import yaml
import logging

import utils
import data
import engine

from vocab import deserialize_vocab


def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSICD_JGDN.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--resume', default='checkpoint/rsicd_dual_SAF_a1.0/0/AMFMN_best.pth.tar', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    options['optim']['resume'] = opt.resume

    return options

def main(options):
    # choose model
    if options['name'] == "AMFMN":
        from layers import JGDN as models
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # Create dataset, model, criterion and optimizer
    test_loader = data.get_test_loader(vocab, options)

    model = models.factory(options,
                           vocab_word,
                           cuda=True,
                           data_parallel=False)

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'])
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))

    # evaluate on test set
    sims = engine.validate_test(test_loader, model)

    return sims

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" \
                                         + str(k) + "/" + options['model']['name'] + '_best.pth.tar'

    return updated_options

if __name__ == '__main__':
    options = parser_options()

    # run experiment
    one_sims = main(options)

    # ave
    last_sims = one_sims

    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(last_sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(last_sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )

    print(all_score)
