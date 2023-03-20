import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import pdb
import math
import argparse
import sys
import numpy as np
import datetime
from torch.nn.parameter import Parameter
import time
from .resnet10F import ResNet, SimpleBlock
sys.dont_write_bytecode = True


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    #if classname.find('Conv') != -1:
    if classname.find('Conv2d') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    #if classname.find('Conv') != -1:
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    #if classname.find('Conv') != -1:
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    #if classname.find('Conv') != -1:
    if classname.find('Conv2d') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def gem(x, p=3, eps=1e-6):
    p = Parameter(torch.ones(1)*p).cuda()
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def powerlaw(x, eps=1e-6):
    x = x + self.eps
    return x.abs().sqrt().mul(x.sign())

class Conv_64F(nn.Module):
    def __init__(self, P, Q, norm_layer=nn.BatchNorm2d, neighbor_k=1 ):
        super(Conv_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.hidden = nn.Linear(3, 3)
        self.output_sum = nn.Linear(3,  1)
        self.P = P
        self.Q = Q
        self.similarity = MML_Metric(hidden3=self.hidden, output=self.output_sum, topP=self.P, topQ=self.Q, neighbor_k=neighbor_k)


    def MML_loss(self, q):   
        q = q.contiguous()
        q = q.view(q.size(0), q.size(1), -1)
        q = q.permute(0, 2, 1)
        similar = self.similarity(q, q)

        return similar


    def forward(self, input1):
        q = input1.contiguous()
        q_fc = q.view(q.size(0),-1)

        return q_fc


def define_FewShotNet(P, Q, which_model='Conv64',norm='batch',init_type='normal',  **kwargs):
    FewShotNet = None
    norm_layer = get_norm_layer(norm_type=norm)


    if which_model == 'Conv64F':
        FewShotNet = Conv_64F(P, Q, norm_layer=norm_layer, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)

    init_weights(FewShotNet, init_type=init_type)
    

    return FewShotNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class MML_Metric(nn.Module):
    def __init__(self, hidden3, output, topP, topQ, neighbor_k=1):
        super(MML_Metric, self).__init__()
        self.neighbor_k = neighbor_k

        self.hidden = hidden3
        self.output_sum = output
        self.topP = topP
        self.topQ = topQ


    def cal_MML_similarity(self, input1_batch, input2_batch):

        #Similarity = 0
        Similarity_list = []
        start = time.time()
      
        q_fc = input1_batch.contiguous().view(input1_batch.size(0),-1)
        norm = q_fc.norm(dim=1, p=2, keepdim=True)
        q_fc = q_fc.div(norm)
        cos_sim = q_fc.mm(q_fc.t())

        input1_norm = torch.norm(input1_batch, 2, 2, True)
        input2_norm = torch.norm(input2_batch, 2, 1, True)
        
        query_norm_l = input1_batch / input1_norm 
        support_norm_l = query_norm_l 
        query_norm_p = input2_batch / input2_norm  
        support_norm_p = query_norm_p 
        #assert (torch.min(input1_norm) > 0)
        #assert (torch.min(input2_norm) > 0)

        support_norm_l = support_norm_l.permute(0, 2, 1) 
        support_norm_l = support_norm_l.contiguous().view(-1, support_norm_l.size(1), support_norm_l.size(2))
        
        innerproduct_matrix_l = torch.matmul(query_norm_l.unsqueeze(1),support_norm_l) 
        innerproduct_matrix_p = torch.matmul(query_norm_p.permute(0, 2, 1).unsqueeze(1), support_norm_p)

        topk_value_l, topk_index_l = torch.topk(innerproduct_matrix_l, self.neighbor_k, 3)
        sum3 = torch.sum(topk_value_l, 3)  
        topk_value_l, topk_index_l = torch.topk(sum3, self.topP, 2)
        local = torch.sum(topk_value_l, 2)/self.topP 

        topk_value_p, topk_index_p = torch.topk(innerproduct_matrix_p, self.neighbor_k, 3)
        sum3 = torch.sum(topk_value_p, 3) 
        topk_value_p, topk_index_p = torch.topk(sum3, self.topQ, 2)
        part = torch.sum(topk_value_p, 2)/self.topQ 

    
        a = torch.reshape(local, (-1,1))
        b = torch.reshape(part,(-1,1))
        c = torch.reshape(cos_sim,(-1,1))
        new = torch.cat((a,b,c),dim=1)

        x = F.relu(self.hidden(new))
        out = self.output_sum(x)
        #print(out.shape, out)
        kl_sim_soft = torch.reshape(out, (36, 36))
        kl_sim_soft = torch.softmax(kl_sim_soft, dim=1)

        return kl_sim_soft

    def forward(self, x1, x2):

        Similarity_list = self.cal_MML_similarity(x1, x2)
        
        return Similarity_list

