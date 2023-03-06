import torch
import torch.nn as nn
import numpy as np
import tqdm

from transformers import BertModel, BertConfig, BertTokenizer


class NumNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(NumNet, self).__init__()

        self.modelConfig = BertConfig.from_pretrained('/data/data_yl/modeling_bert/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained(
            '/data/data_yl/modeling_bert/bert-base-uncased-pytorch_model.bin', config=self.modelConfig)
        self.tokenizer = BertTokenizer.from_pretrained('/data/data_yl/modeling_bert/bert-base-uncased-vocab.txt')

        self.embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(self.embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, text):
        tokens, segments, input_masks = process_text(self.tokenizer, text)
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features

# ——————输入处理——————
def process_text(tokenizer, text):
    tokens, segments, input_masks = [], [], []
    tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])  # 最大的句子长度


    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)
    return tokens_tensor, segments_tensors, input_masks_tensors

class MergeNum(nn.Module):
    def __init__(self, dim, weight_num=0.9):
        super(MergeNum, self).__init__()
        self.dim = dim
        self.num_encoder = NumNet(dim)
        self.weight_num = weight_num
        self.nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                     "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

    def forward(self, x):
        num = x.shape[0]
        if num <= 20:
            num_feature = self.num_encoder(self.nums[num])
            out = (x.sum(0) / num) + num_feature * self.weight_num
        else:
            out = x.sum(0) / num
        return out


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_sim(fea):
    """Cosine similarity between all the image and sentence pairs
    """
    fea = l2norm(fea)
    return fea.mm(fea.t())


def find_intersection(s):
    for i, v in enumerate(s):
        for j, k in enumerate(s[i + 1:], i + 1):
            if v & k:
                s[i] = v.union(s.pop(j))
                return find_intersection(s)
    return s


def get_sim_pairs(features, limit: float):
    # features； 36 * 2048
    # return pair： (x, y)
    sims = cosine_sim(features)
    # 将sims矩阵下半角设为0
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            if i >= j:
                sims[i][j] = 0.
    # 得到大于阈值的相似度的下标
    pairs_torch = torch.where(sims >= limit)
    pair_x, pair_y = pairs_torch
    num = pair_x.shape[0]
    pairs = None
    if 0< num <= 20:
        pairs = []
        for i in range(num):
            pairs.append((pair_x[i].item(), pair_y[i].item()))
        pairs = [set(i) for i in pairs if i]
        pairs = find_intersection(pairs)
    return pairs


def merge_feature(unmerged_feature, num_merge):
    return num_merge(unmerged_feature)


def merge_pair_feature(features, limit, num_merge):
    """
    merge all the pair features
    """
    pairs = get_sim_pairs(features, limit)
    lens = features.shape[0]
    dim = features.shape[1]
    lens_pair = 0
    if pairs is not None:
        # 初始化合成特征
        merged_features = torch.zeros(lens, dim)
        # 将原始特征赋值
        merged_features[:lens, :] = features
        # 合成特征
        for i in range(len(pairs)):
            pair = list(pairs[i])
            unmerged_feature = features[pair]
            merged_feature = merge_feature(unmerged_feature, num_merge)
            merged_features[pair] = merged_feature

        lens_pair = len(pairs)
    else:
        merged_features = features
    return merged_features, lens_pair

def merge_batch_features(batch_features, visual_lens, limit, num_merge):
    num_features = batch_features.shape[0]
    lens = batch_features.shape[1]
    dim = batch_features.shape[2]
    features = batch_features[0]
    merged_features, num_merge_feature = merge_pair_feature(features, limit, num_merge)
    return merged_features, visual_lens


if __name__ == '__main__':
    ######################################################
    # path_att2 = "/data/data_yl/TERAN/f30k/images/features_36/bu_att/0.npz"
    # bu_att2 = np.load(path_att2)['feat']
    # bu_att2 = torch.from_numpy(bu_att2)
    # bu_att2 = bu_att2.unsqueeze(0)
    # print(bu_att2.shape)
    #
    # num_merge = MergeNum(2048, weight_num=0.9)
    # merged_batch_features, visual_lens = merge_batch_features(bu_att2, [bu_att2.shape[1]], 0.9, num_merge)
    # print(merged_batch_features.squeeze(0).shape)
    # print(visual_lens)
    import os
    import sys
    path_att_dir = "/data/data_yl/TERAN/f30k/images/features_36/bu_att/"
    path_new_att_dir = "/data/data_yl/TERAN/f30k/images/features_36/bu_att_new/"
    atts = os.listdir(path_att_dir)
    num_merge = MergeNum(2048, weight_num=1)
    max_lens = 0
    for att in tqdm.tqdm(atts):
        path_att = path_att_dir + att
        bu_att = np.load(path_att)['feat']
        bu_att = torch.from_numpy(bu_att)
        bu_att = bu_att.unsqueeze(0)
        merged_batch_features, visual_lens = merge_batch_features(bu_att, [bu_att.shape[1]], 0.9, num_merge)
        if visual_lens[0] > max_lens:
            max_lens = visual_lens[0]
            print(max_lens)
        np.savez(path_new_att_dir + att, feat=merged_batch_features.squeeze(0).cpu().detach().numpy())
    # data = np.load(path_new_att_dir + "10133.npz")
    # print(data['feat'].shape)
