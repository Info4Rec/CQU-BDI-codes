import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

 
class weight_nd_Loss(nn.Module):
    
    def __init__(self,batch_size,instances):
        super(weight_nd_Loss,self).__init__()
        self.batch_size = batch_size
        self.instances = instances
        assert batch_size%instances == 0
        cls = batch_size//instances 
        eye = torch.eye(cls).view((cls,cls,1)).byte()
        device = torch.device('cuda:0')
        self.pos_mask = eye.repeat((1, 1, instances)).view((cls, 1, batch_size)).repeat((1, instances, 1)).view((batch_size, batch_size))
        self.pos_mask = self.pos_mask & torch.triu(torch.ones_like(self.pos_mask), 1)
        self.pos_mask = self.pos_mask.to(device)
        self.neg_mask = torch.triu(torch.ones_like(self.pos_mask), 1)
        self.neg_mask[self.pos_mask] = 0 
        self.neg_mask = self.neg_mask.to(device)
        
    
    def distance(self,e):
        return -2 * e.mm(torch.t(e)) + e.pow(2).sum(dim=1).view(1,-1) + e.pow(2).sum(dim=1).view(-1,1)

 
    def forward(self,x,y,mean,std):
        cls_mean = mean[y][:,y]
        cls_std = std[y][:,y]
        cls_obs = torch.normal(cls_mean,cls_std)
        dist = self.distance(x)
        zero = torch.zeros_like(dist,device='cuda:0')
        sign_mask_n1 = torch.zeros_like(dist,device='cuda:0')
        sign_mask_n2 = torch.zeros_like(dist,device='cuda:0')
        sign_mask_p1 = torch.zeros_like(dist,device='cuda:0')
        sign_mask_n3 = torch.zeros_like(dist,device='cuda:0')
        margin_n = 2.5
        margin_p = 2
        T = 10

        dist_n1 = (dist < cls_obs- margin_n*cls_std)
        index_n1 = (dist_n1.mul(self.neg_mask>0)).nonzero()
        while ((index_n1.shape == torch.Size([0,2]))&(margin_n>1.98)):
          margin_n = margin_n-0.001
          dist_n1 = (dist < cls_obs- margin_n*cls_std)
          index_n1 = (dist_n1.mul(self.neg_mask>0)).nonzero()

        if((index_n1.shape == torch.Size([0,2]))&(margin_n<1.98)):
          loss_n1 = -10e-5     
          
        if (index_n1.shape!=torch.Size([0, 2])):
          for i in range(len(index_n1)):
              sign_mask_n1[index_n1[i][0]][index_n1[i][1]] = torch.exp(T*(cls_obs[index_n1[i][0]][index_n1[i][1]]- margin_n*cls_std[index_n1[i][0]][index_n1[i][1]]-dist[index_n1[i][0]][index_n1[i][1]]))

          dist_1 = torch.where(dist < cls_obs- margin_n*cls_std,dist,zero)
          margin_1 = cls_obs- margin_n*cls_std
          sum_weight_n1 = sign_mask_n1.sum()
          sign_mask_n1 = -sign_mask_n1/sum_weight_n1
          loss_n1 = (sign_mask_n1*dist_1).sum()

        dist_n2 = ((dist > cls_obs- margin_n*cls_std) & (dist < cls_obs- (margin_n-1)*cls_std))
        index_n2 = (dist_n2.mul(self.neg_mask>0)).nonzero()
        if(index_n2.shape == torch.Size([0,2])):
          loss_n2 = -10e-5
        if (index_n2.shape!=torch.Size([0,2])):

          for i in range(len(index_n2)):

            sign_mask_n2[index_n2[i][0]][index_n2[i][1]] = torch.exp((T-5)*(cls_obs[index_n2[i][0]][index_n2[i][1]]-(margin_n-1)*cls_std[index_n2[i][0]][index_n2[i][1]]-dist[index_n2[i][0]][index_n2[i][1]]))


          dist_2 = torch.where(((dist > cls_obs-margin_n*cls_std) & (dist < cls_obs- (margin_n-1)*cls_std)),dist,zero)
          margin_2 = cls_obs - (margin_n-1)*cls_std
          sum_weight_n2 = sign_mask_n2.sum()
          sign_mask_n2 = -sign_mask_n2/sum_weight_n2
          loss_n2 = (sign_mask_n2*dist_2).sum()


        dist_n3 = ((dist > cls_obs-(margin_n-1)*cls_std) & (dist < cls_obs+(margin_n-1)*cls_std))
        index_n3 = (dist_n3.mul(self.neg_mask>0)).nonzero()

        if(index_n3.shape == torch.Size([0,2])):
          loss_n3 = -10e-5

        if (index_n3.shape!=torch.Size([0,2])):

          for i in range(len(index_n3)):

            sign_mask_n3[index_n3[i][0]][index_n3[i][1]] = torch.exp((T-5)*(cls_obs[index_n3[i][0]][index_n3[i][1]]+(margin_n-1)*cls_std[index_n3[i][0]][index_n3[i][1]]-dist[index_n3[i][0]][index_n3[i][1]]))


          dist_3 = torch.where(((dist > cls_obs-(margin_n-1)*cls_std) & (dist < cls_obs+(margin_n-1)*cls_std)),dist,zero)
          margin_3 = cls_obs + (margin_n-1)*cls_std
          sum_weight_n3 = sign_mask_n3.sum()
          sign_mask_n3 = -sign_mask_n3/sum_weight_n3
          loss_n3 = (sign_mask_n3*dist_3).sum()

      

        dist_p1 = (dist > cls_obs+ margin_p*cls_std) 
        index_p1 = (dist_p1.mul(self.pos_mask>0)).nonzero()
        

        while ((index_p1.shape == torch.Size([0,2]))&(margin_p>1.98)):
           margin_p = margin_p-0.001
           dist_p1 = (dist > cls_obs+ margin_p*cls_std) 
           index_p1 = (dist_p1.mul(self.pos_mask>0)).nonzero()

        if((index_p1.shape == torch.Size([0,2]))&(margin_p<1.98)):
           loss_p1 = 10e-5  

        if (index_p1.shape!=torch.Size([0,2])):

          sum_p1 = 0
          filter_p1 = torch.where((dist > cls_obs+ margin_p*cls_std).mul(self.pos_mask>0),dist,zero)
          sum_p1 = filter_p1.sum()

          for l in range(len(index_p1)):
            sign_mask_p1[index_p1[l][0]][index_p1[l][1]] = dist[index_p1[l][0]][index_p1[l][1]]/sum_p1
          
          margin_4 = cls_obs+ margin_p*cls_std
          loss_p1 = (sign_mask_p1*filter_p1).sum()

        
        losssum = loss_n1  + loss_n2 + loss_n3 + loss_p1 
        loss = 1.0 + losssum.div(4)
        
        return dist,loss

