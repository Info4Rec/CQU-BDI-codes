
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from operator import is_
import torch
import sklearn
from sklearn.metrics import silhouette_score, silhouette_samples
import faiss
import numpy as np
import os
import time
import datetime
from torch import autograd
from PIL import ImageFile
import sys
import torch.nn.functional as F
import scipy.stats
sys.dont_write_bytecode = True

import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from logger import setup_logger

from config import cfg
from datasets import build_data
from logger import setup_logger
from models import feat_extractor


def cal_silhouette(X, labels):
    score = silhouette_samples(X, labels, metric='euclidean')
    print(score)
    return score

def distance(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def distance2(a, b):
    dist = -2 * a.matmul(torch.t(b)) + a.pow(2).sum(dim=1).view(1,-1)+ b.pow(2).sum(dim=1).view(1,-1)
    return dist

def euclidean_dist_rank(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def rank_loss_my(dist_mat, labels, global_sim, theta1, theta2, similar_loss):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    sorted_similar, indice = torch.sort(similar_loss,descending=True)
    j,rank = indice.sort(dim=1)
  
    total_loss = 0.0
    indice_losses = 0.0

    for ind in range(N):
      start = time.time()
     
      is_pos = (labels == labels[ind])
      is_neg = ~ is_pos    

      is_pos[ind] = 0      
      similar_loss_ap = similar_loss[ind][is_pos]
      similar_loss_an = similar_loss[ind][is_neg]
      dist_ap = dist_mat[ind][is_pos]
      dist_an = dist_mat[ind][is_neg]


      similar_sum = torch.sum(similar_loss_ap)
      ap_sum = torch.mean(similar_loss_ap)
      an_sum = torch.mean(similar_loss_an)
      loss1 = -1.0*torch.log(ap_sum)

      similar = torch.mean(similar_loss_ap)
      similar_loss_ap[:] = similar
      global_sim = global_sim.detach()   

      if global_sim[labels[ind]][labels[ind]]== 0:
        global_sim[labels[ind]][labels[ind]] = similar
      else:
        global_sim[labels[ind]][labels[ind]] = 0.7 * global_sim[labels[ind]][labels[ind]] + 0.3 * similar
      similar_loss_ap[:] = global_sim[labels[ind]][labels[ind]]

      for label in set(labels[is_neg].data.cpu().numpy()):
        a = (labels[is_neg] == label)
        similar = torch.mean(similar_loss_an[a])

        if global_sim[labels[ind]][label] == 0:
          global_sim[labels[ind]][label] = similar
        else:
          global_sim[labels[ind]][label] = 0.7 * global_sim[labels[ind]][label] + 0.3 * similar
          
        similar_loss_an[a] = global_sim[labels[ind]][label]  

  
      ap_margin = theta1 - torch.exp(similar_loss_ap) #1.6
      ap_mask = torch.ge(dist_ap, ap_margin)
      size_pos = dist_ap[ap_mask].size(0)

      if size_pos==0:
        loss_ap = 1e-5
      else:
        ap_weight = dist_ap[ap_mask]
        ap_weight_sum = torch.sum(ap_weight)
        loss_ap = torch.sum( torch.mul(ap_weight, dist_ap[ap_mask] - ap_margin[ap_mask]) )
        loss_ap = torch.div(loss_ap, ap_weight_sum)
  
      an_margin = theta2 - torch.exp(similar_loss_an) #1.8    
      an_mask = torch.ge(an_margin, dist_an)
      size_neg = dist_an[an_mask].size(0)

      if size_neg==0:
        loss_an = 1e-5
      else: 
        an_weight = an_margin[an_mask]-dist_an[an_mask]
        an_weight_sum = torch.sum(an_weight)
        loss_an = torch.sum( torch.mul(an_weight, an_margin[an_mask] - dist_an[an_mask]) )
        loss_an = torch.div(loss_an,an_weight_sum)
      
      #print("loss_ap,loss_an,loss1:",loss_ap, loss_an, loss1)   
      total_loss = total_loss + loss_ap + loss_an + 0.5*loss1
      

    total_loss = total_loss*1.0
    return total_loss, global_sim



class Ranked_Loss(object):
    
    def __init__(self, theta1, theta2):
        #super(RankedLoss, self).__init__()
        self.theta1 = theta1
        self.theta2 = theta2
        
    def __call__(self, global_feat, labels, global_similarity_matrix, similar_list, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
           # global_trans = normalize_rank(global_trans, axis=-1)

        #dist_mat = euclidean_dist_rank(global_feat, global_trans)
        dist_mat = distance2(global_feat, global_feat)
        dist_mat = dist_mat.cuda()
        total_loss, global_sim = rank_loss_my(dist_mat, labels, global_similarity_matrix, self.theta1, self.theta2, similar_list)
        
        return total_loss, global_sim

