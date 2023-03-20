
def cross_loss():
    criterion = nn.CrossEntropyLoss().cuda()
    cross_loss = 0
    targets = targets.cuda()
    print(targets)

    arr = torch.arange(0,60).cuda()
    
    for i in range(len(similar_loss)):
        #print("similar_loss:", similar_loss[i].shape, arr.shape)#[60,60],[60]
        temp_loss = criterion(similar_loss[i], arr)
        cross_loss = cross_loss + temp_loss

def log_loss():
    #=====================log loss=================
    dist = dist_func(ori_feature, trans_feature)
    print("dist shape:",dist.shape)
    print("torch.div(dist, similar_loss)",torch.div(dist, similar_loss))
    loss_log = torch.sum( torch.log(torch.div(dist, similar_loss)) )


def clustering_loss():
    #=====================Clustering loss=======================
    #把原图和空间变换后的图像都进行聚类
    ori_feature = torch.cat((ori_feature, trans_feature), 0) #[240,512]
    targets = torch.cat((targets, targets), 0)

    ori_feature = ori_feature.contiguous().view(ori_feature.size(0), -1).cpu().detach().numpy()
    print(ori_feature.shape) #[128,1024]
    #print(targets.shape) #[128]

    kmeans = faiss.Kmeans(cfg.MODEL.HEAD.DIM, 8)#30个类!!!!!
    kmeans.train(ori_feature)
    D, I = kmeans.index.search(ori_feature, 2)#I：每一行向量对应的最接近的聚类centroid，D：对应的平方L2距离
    length = len(D)
    D1, I1 = kmeans.index.search(ori_feature, 1)
    label1 = np.squeeze(I1)
    cluster_centers = kmeans.centroids #聚类后的聚类中心
    #print("D,I:",D.shape, I.shape)
    kmean_generate_label_CUB(cfg.DATA.TRAIN_IMG_ORI, label1)
    #print("聚类后的聚类中心：", cluster_centers.shape)#[16,28224]


class clustering_level_loss(nn.Module):
    def __init__(self, margin):
        self.margin = margin
        super(clustering_level_loss, self).__init__()


    def forward(self, embeddings, cluster_centers):
        #进一步将原图fi与变换后图像fi'拉近，与不属于的聚类中心拉远
        cluster_centers = torch.from_numpy(cluster_centers)
        all_dis = torch.mm(ori_feature, cluster_centers.t())

        #归一化
        #net2 = nn.Softmax(dim = 1)
        #all_dis = net2(all_dis)
        #all_dis = F.pairwise_distance(ori_feature, cluster_centers, p=2)
        #print("all_dis shape:", all_dis.shape)#[256,16]
        #print("all_dis:",all_dis)
        dis_sum = all_dis.sum(1)
        #print("dis_sum shape:",dis_sum.shape) #[256]
        cluster_level_loss = torch.tensor(0.).cuda()
        cluster_level_loss.requires_grad = True
        for i in range(len(dis_sum)):
            cluster_p = all_dis[i][label1[i]]
            cluster_n = dis_sum[i] - cluster_p

            cluster_loss = 1 + (cluster_n - cluster_p).div(dis_sum[i])
            cluster_level_loss = cluster_level_loss + cluster_loss


        return cluster_level_loss