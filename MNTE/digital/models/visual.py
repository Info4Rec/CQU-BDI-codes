from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from .utils import PositionalEncodingImageBoxes, l2norm


def EncoderImage(opt):

    embed_size = opt.embed_size

    transformer_layers = opt.image_transformer_layers
    pos_encoding = opt.pos_encoding
    visual_feat_dim = opt.image_feat_dim
    dropout = opt.dropout
    img_enc = TransformerPostProcessing(transformer_layers, visual_feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=pos_encoding, dropout=dropout)

    return img_enc


class TransformerPostProcessing(nn.Module):
    def __init__(self, num_transformer_layers, feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=None, dropout=0.1):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                         num_layers=num_transformer_layers)
        if pos_encoding is not None:
            self.pos_encoding_image = PositionalEncodingImageBoxes(feat_dim, pos_encoding)
        self.fc = nn.Linear(feat_dim, embed_size)
        self.aggr = aggr
        if aggr == 'gated':
            self.gate_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 1)
            )
            self.node_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        self.pos_encoding = pos_encoding

    def forward(self, visual_feats, visual_feats_len=None, boxes=None):

        visual_feats = visual_feats.permute(1, 0, 2)
        if self.pos_encoding is not None:
            visual_feats = self.pos_encoding_image(visual_feats, boxes)

        if visual_feats_len is not None:
            bs = visual_feats.shape[1]
            # construct the attention mask
            max_len = max(visual_feats_len)
            mask = torch.zeros(bs, max_len).bool()
            for e, l in zip(mask, visual_feats_len):
                e[l:] = True
            mask = mask.to(visual_feats.device)
        else:
            mask = None

        visual_feats = self.transformer_encoder(visual_feats, src_key_padding_mask=mask)
        # visual_feats = visual_feats.permute(1, 0, 2)

        if self.aggr == 'mean':
            out = visual_feats.mean(dim=0)
        elif self.aggr == 'gated':
            out = visual_feats.permute(1, 0, 2)
            m = torch.sigmoid(self.gate_fn(out))   # B x S x 1
            v = self.node_fn(out)   # B x S x dim
            out = torch.bmm(m.permute(0, 2, 1), v)      # B x 1 x dim
            out = out.squeeze(1)    # B x dim
        else:
            out = visual_feats[0]

        out = self.fc(out)

        return out, visual_feats.permute(1, 0, 2)


def find_nhead(feat_dim, higher=8):
    # find the right n_head value (the highest value lower than 'higher')
    for i in reversed(range(higher + 1)):
        if feat_dim % i == 0:
            return i
    return 1


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)