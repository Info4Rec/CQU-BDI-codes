import torch
import torch.nn as nn
from basic_model import googlenet
from losses.get_loss import get_loss

from . import nd_loss



class get_feature(nn.Module):
    def __init__(self,pretrained=True):
        super(get_feature,self).__init__()
        self.google_net = googlenet.googlenet(pretrained=pretrained)

    def forward(self,x):
        if self.google_net.training and self.google_net.aux_logits:
            _, _, outputs = self.google_net(x)
        else:
            outputs = self.google_net(x)
        #outputs = self.google_net(x)
        return outputs

class get_embedding(nn.Module):
    def __init__(self,embedding_size,dim=1000):
        super(get_embedding,self).__init__()
        self.dim = dim
        self.linear = nn.Linear(1000, dim)
    

    def forward(self,x):
        x = self.linear(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        embedding_x = x.div(norm)
        return embedding_x 



def MeanStd(dist,y,n_ins,bs,last_delta_mean,last_delta_std):
    beta_m = 0.8 if (last_delta_mean is not None) else 0
    beta_s = 0.8 if (last_delta_std is not None) else 0
    y = y.unique().flip(0)
    cls_last_delta_mean = last_delta_mean[y][:,y]
    cls_last_delta_std = last_delta_std[y][:,y]
    assert bs % n_ins == 0
    cls = bs // n_ins
    group = dist.view(cls, n_ins, bs).transpose(2, 1).view(cls, cls, n_ins, n_ins).flatten(-2)
    cls_last_delta_mean = beta_m*cls_last_delta_mean + (1-beta_m)*group.mean(-1)
    cls_last_delta_std = beta_s*cls_last_delta_std + (1-beta_s)*group.std(-1)

    return last_delta_mean,last_delta_std
        

class NDfdml(nn.Module):
    def __init__(self,n_class,batch_size,instances,embedding_size=128,pretrained=True):
        super(NDfdml,self).__init__()
        device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.instances = instances
        assert batch_size%instances == 0
        self.n_class = batch_size//instances
        self.googlelayer = get_feature(pretrained).to(device)
        self.embedding_layer = get_embedding(dim=1000,embedding_size=embedding_size).to(device)
        self.dataset_metricloss = get_loss('Triplet')
        self.loss_fn = nd_loss.weight_nd_Loss(self.batch_size,self.instances)
        
        

    def forward(self,x,y,mean,std,last_delta_mean,last_delta_std,it):
        feature_x = self.googlelayer(x)
        embedding_x = self.embedding_layer(feature_x)
        jm ,_ = self.dataset_metricloss(embedding_x,y)
        dist,loss_np = self.loss_fn(embedding_x,y,mean,std)
        loss = jm + loss_np
        
        last_delta_mean_new,last_delta_std_new = MeanStd(dist,y,self.instances,self.batch_size,last_delta_mean,last_delta_std)
        
        return last_delta_mean_new,last_delta_std_new,loss






    


