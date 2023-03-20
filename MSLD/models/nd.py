import torch
import torch.nn as nn
from basic_model import M2L_KL_ori, BNInception
import torchvision.models as models
import torch.nn.functional as F
from losses.RankedList_loss import Ranked_Loss
from basic_model.M2L_KL_ori import weights_init_kaiming, weights_init_kaiming2, weights_init_classifier


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

class get_feature(nn.Module):
    def __init__(self,pretrained=False):
        super(get_feature,self).__init__()       
        self.google_net = BNInception.bninception(num_classes=1000)
        net_dict = self.google_net.state_dict()
        predict_model = torch.load('./basic_model/bn_inception-52deb4733.pth')
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        self.google_net.load_state_dict(net_dict) 


    def forward(self,x):
    
        outputs, outputs_local = self.google_net(x)
  
        return outputs, outputs_local


class get_similar_list(nn.Module):
    def __init__(self, P, Q):
        super(get_similar_list,self).__init__()
        self.P = P
        self.Q = Q
        self.conv64 = M2L_KL_ori.define_FewShotNet(self.P, self.Q, which_model='Conv64F', neighbor_k=1, norm='batch', init_type='normal')  

    def forward(self,x):
        outputs = self.conv64(x)

        return outputs

    def cal_similar(self, x):
        similar_list = self.conv64.MML_loss(x)

        return similar_list


class get_embedding(nn.Module):
    def __init__(self,dim=512):
        super(get_embedding,self).__init__()
        self.dim = dim
        self.linear = nn.Linear(1024, dim)      
        self.linear.apply(weights_init_kaiming)


    def forward(self,x):
        x = self.linear(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        embedding_x = x.div(norm)
        return embedding_x 


def dist_func_2(a, b):
    dist = -2 * a.matmul(torch.t(b)) + a.pow(2).sum(dim=1).view(1,-1)+ b.pow(2).sum(dim=1).view(1,-1)
    return dist


class NDfdml(nn.Module):
    def __init__(self,pretrained=True, theta1=1.6, theta2=1.7, P=5, Q=100):
        super(NDfdml,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theta1 = theta1
        self.theta2 = theta2
        self.P = P
        self.Q = Q
        self.googlelayer = get_feature(pretrained).to(self.device)
        self.conv64layer = get_similar_list(self.P, self.Q).to(self.device)
        self.embedding_layer = get_embedding(dim=512).to(self.device)
        self.RL = Ranked_Loss(theta1=self.theta1, theta2=self.theta2) 

    def forward(self,x, y, global_similarity_matrix):

        feature_x, feature_x_local = self.googlelayer(x)
        embedding = self.embedding_layer(feature_x)
        #print("embedding shape:", embedding.shape)

        similar_list = self.conv64layer.cal_similar(feature_x_local) 
        clustering_loss3, global_sim = self.RL(embedding, y, global_similarity_matrix, similar_list=similar_list, normalize_feature=True)

        return clustering_loss3, global_sim






    


