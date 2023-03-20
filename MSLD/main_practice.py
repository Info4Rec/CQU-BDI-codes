import argparse
import torch
import faiss
import numpy as np
import datetime
import time
import math
import re
import random
import os
from torch import autograd
from PIL import ImageFile
import pdb
import sys
import util
sys.dont_write_bytecode = True

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from config import cfg
from logger import setup_logger
from utils.metric_logger import MetricLogger

# ============================ Data & Networks =====================================
from datasets import build_transforms,build_data
from test2 import test
from models import feat_extractor
from models import nd
from torch.optim import lr_scheduler

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cfg.cuda = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = False

torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
np.random.seed(1024)

cfg.outf, F_txt = util.set_save_path(cfg)
logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)


model = nd.NDfdml(pretrained=True, theta1=cfg.theta1, theta2=cfg.theta2, P=cfg.P, Q=cfg.Q)
# ========================================================= Define functions ============================================================
def test_recall(model, val_loader):
    print("test_loader:",len(val_loader))
    test_feature = feat_extractor(model, val_loader, cfg, logger=logger)
    test_labels = val_loader.dataset.label_list
    test_labels = np.array([int(k) for k in test_labels])

    np.save(os.path.join(cfg.outf, 'car_test_feat.npy'), test_feature)
    np.save(os.path.join(cfg.outf, 'car_test_labels.npy'), test_labels)              
    print("test_feature shape:",test_feature.shape) #[5924,1000]
    #test_feature = test_feature.contiguous().view(test_feature.size(0), -1).numpy()                

    get_metrics = test(test_feature,test_labels)

    return get_metrics

def train():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global model
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model, device_ids=[0,1])    
        
    print(cfg, file=F_txt)
    model = model.to(device)
    
    global best_prec1_val, best_prec1_test, epoch_index
    best_prec1_val = 0
    best_prec1_test = 0
    epoch_index = 0

    #=====================define loss function (criterion) and optimizer========== 
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    
    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            epoch_index = checkpoint['epoch_index']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch_index']))
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch_index']), file=F_txt)
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
            print("=> no checkpoint found at '{}'".format(cfg.resume), file=F_txt)


    #train_loader = build_data(cfg, is_train=2) #cfg.DATA.TRAIN_IMG_SOURCE,transforms
    #logger.info(train_loader.dataset) #???     
    val_loader = build_data(cfg, is_train=0)
    logger.info(val_loader.dataset)
    train_val_loader = build_data(cfg, is_train=1)
    logger.info(train_val_loader.dataset)

    logger.info("======================Start training=========================")
    meters = MetricLogger(delimiter="") 
    
    arguments = dict()
    arguments["iteration"] = 0

    max_iter = len(train_val_loader) #5864
    start_iter = arguments["iteration"]
    best_iteration = -1
    best_recall = 0

    feature_plot, sim_plot, cluster_plot, indice_plot, total_plot = [],[],[],[],[]
    max_r8,max_epoch,max_iteration = 0,0,0
    
    similarity_matrix = torch.zeros(100,100) #(eye)
    similarity_matrix.requires_grad = False
    similarity_matrix = similarity_matrix.to(device)

    for epoch in range(cfg.Epoch):
        print("====================epoch:====================== ", epoch)
        start_training_time = time.time()
        iter = epoch * 200
        model.train()

        for iteration, (images, targets) in enumerate(train_val_loader, start_iter):
            iteration = iter + iteration
            if iteration % 200 == 0 or iteration ==  max_iter:
                logger.info(
                    meters.delimiter.join(
                        ["iter: {iter} ",
                            "{meters}",
                            "lr: {lr:.6f} ",
                            "max mem: {memory:.1f} GB",]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated(device=device) / 1024.0 / 1024.0 / 1024.0,
                    )
                )

                with torch.no_grad():
                    model.eval()
                    #==============================test_loader============================
                    get_metrics = test_recall(model, val_loader)
                    prec1_test = get_metrics['R@1']
                    if prec1_test > max_r8:
                        max_r8 = prec1_test
                        max_epoch = epoch
                        max_iteration = iteration   
                        model_path = os.path.join(cfg.outf, 'model_best_test.pth.tar')
                        torch.save(model.state_dict(), model_path)

                    print("max_r8, max_epoch, max_iteration:",max_r8,max_epoch, max_iteration)

            model.train()            
            input_var1 = images.contiguous().view(-1, images.size(1), images.size(2),  images.size(3))
            input_var1 = input_var1.to(device)
            targets = targets.to(device)      

    
            with torch.set_grad_enabled(True):            
                clustering_loss3, global_sim= model(input_var1, targets, similarity_matrix)
                similarity_matrix = global_sim
                loss = torch.tensor(0.).cuda()
                loss.requires_grad = True
                loss = clustering_loss3
                print("## loss:", loss)

            iteration = iteration + 1
            arguments["iteration"] = iteration

            #=====Compute gradients and do SGD step======
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()     
         
        #total_training_time = time.time() - start_training_time
        #total_time_str = str(datetime.timedelta(seconds=total_training_time))
        #logger.info("Epoch Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter))
        #)
    
    #util.plot_loss_curve_2(cfg, cluster_plot)


def dist_func_2(a, b):
    dist = -2 * np.matmul(a, np.transpose(b)) + np.sum(np.square(a), 1).reshape(1, -1) + np.sum(np.square(b), 1).reshape(-1, 1)
    return dist

def dist_func(a, b):
    dist = -2 * a.matmul(torch.t(b)) + a.pow(2).sum(dim=1).view(1,-1)+ b.pow(2).sum(dim=1).view(1,-1)
    return dist

def euclidean_dist_rank(x, y):
    """
    Args:x: pytorch Variable, with shape [m, d]
         y: pytorch Variable, with shape [n, d]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_elu_dis(data):
    return torch.sqrt((-2*data.mm(data.t()))+torch.sum(torch.square(data),axis=1,keepdim=True)+torch.sum(torch.square(data.t()),axis=0,keepdim=True))

#===========================================main()===============================================
if __name__ == '__main__':

    logger = setup_logger(name='Test', level=cfg.LOGGER.LEVEL)

    if cfg.mode == 'train':
        train()
    else:
        logger.info("Start testing")
        model.eval()
        logger.info('test')

        val_loader = build_data(cfg, is_train=0)
        logger.info(val_loader.dataset)

        model_path = os.path.join(cfg.outf, 'model_best_test.pth.tar')
        if not os.path.exists(model_path):
            raise Exception('Can not find trained model {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))

        model.eval()
        #get_metrics = test_recall(model, val_loader)
        test_feature = feat_extractor(model, val_loader, cfg, logger=logger)
        test_labels = val_loader.dataset.label_list
        test_labels = np.array([int(k) for k in test_labels])

        test_feature = torch.from_numpy(test_feature)
        test_feature = test_feature.contiguous().view(test_feature.size(0), -1).numpy()
        print(test_feature.shape)
        get_metrics = test(test_feature,test_labels)

