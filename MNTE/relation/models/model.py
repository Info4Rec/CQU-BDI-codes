import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .loss import  AlignmentContrastiveLoss
from .text import  EncoderText
from .visual import  EncoderImage

from .utils import l2norm, Aggregator


class JointTextImageTransformerEncoder(nn.Module):
    """
    This is a bert caption encoder - transformer image encoder (using bottomup features).
    If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
    """

    def __init__(self, opt):
        super().__init__()
        self.txt_enc = EncoderText(opt)

        visual_feat_dim = opt.image_feat_dim
        caption_feat_dim = opt.text_word_dim
        dropout = opt.dropout
        layers = opt.layers
        embed_size = opt.embed_size
        self.img_enc = EncoderImage(opt)

        self.img_proj = nn.Linear(visual_feat_dim, embed_size)
        self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
        self.embed_size = embed_size
        self.shared_transformer = opt.shared_transformer

        transformer_layer_1 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                         dim_feedforward=2048,
                                                         dropout=dropout, activation='relu')
        self.transformer_encoder_1 = nn.TransformerEncoder(transformer_layer_1,
                                                           num_layers=layers)
        if not self.shared_transformer:
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                             dim_feedforward=2048,
                                                             dropout=dropout, activation='relu')
            self.transformer_encoder_2 = nn.TransformerEncoder(transformer_layer_2,
                                                               num_layers=layers)
        self.text_aggregation = Aggregator(embed_size, aggregation_type=opt.text_aggregation)
        self.image_aggregation = Aggregator(embed_size, aggregation_type=opt.image_aggregation)
        self.text_aggregation_type = opt.text_aggregation
        self.img_aggregation_type = opt.image_aggregation

    def forward(self, features, captions, feat_len, cap_len, boxes):
        # process captions by using bert
        full_cap_emb_aggr, c_emb = self.txt_enc(captions, cap_len)  # B x S x cap_dim

        # process image regions using a two-layer transformer
        full_img_emb_aggr, i_emb = self.img_enc(features, feat_len, boxes)  # B x S x vis_dim
        # i_emb = i_emb.permute(1, 0, 2)                             # B x S x vis_dim

        bs = features.shape[0]

        # forward the captions
        if self.text_aggregation_type is not None:
            c_emb = self.cap_proj(c_emb)

            mask = torch.zeros(bs, max(cap_len)).bool()
            mask = mask.to(features.device)
            for m, c_len in zip(mask, cap_len):
                m[c_len:] = True
            full_cap_emb = self.transformer_encoder_1(c_emb.permute(1, 0, 2),
                                                      src_key_padding_mask=mask)  # S_txt x B x dim
            full_cap_emb_aggr = self.text_aggregation(full_cap_emb, cap_len, mask)
        # else use the embedding output by the txt model
        else:
            full_cap_emb = None

        # forward the regions
        if self.img_aggregation_type is not None:
            i_emb = self.img_proj(i_emb)

            mask = torch.zeros(bs, max(feat_len)).bool()
            mask = mask.to(features.device)
            for m, v_len in zip(mask, feat_len):
                m[v_len:] = True
            if self.shared_transformer:
                full_img_emb = self.transformer_encoder_1(i_emb.permute(1, 0, 2),
                                                          src_key_padding_mask=mask)  # S_txt x B x dim
            else:
                full_img_emb = self.transformer_encoder_2(i_emb.permute(1, 0, 2),
                                                          src_key_padding_mask=mask)  # S_txt x B x dim
            full_img_emb_aggr = self.image_aggregation(full_img_emb, feat_len, mask)
        else:
            full_img_emb = None

        full_cap_emb_aggr = l2norm(full_cap_emb_aggr)
        full_img_emb_aggr = l2norm(full_img_emb_aggr)

        # normalize even every vector of the set
        full_img_emb = F.normalize(full_img_emb, p=2, dim=2)
        full_cap_emb = F.normalize(full_cap_emb, p=2, dim=2)


        return full_img_emb_aggr, full_cap_emb_aggr, full_img_emb, full_cap_emb


class Relation(torch.nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(opt)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.alignment_criterion = AlignmentContrastiveLoss(margin=opt.margin,
                                                            measure=opt.measure,
                                                            max_violation=opt.max_violation)

        self.Eiters = 0
        self.opt = opt


    def forward_emb(self, images, captions, img_len, cap_len, boxes):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            boxes = boxes.cuda()

        # Forward
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.img_txt_enc(images, captions, img_len, cap_len, boxes)

        return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_len

    def get_parameters(self):
        lr_multiplier = 1.0 if self.opt.text_finetune else 0.0

        ret = []
        params = list(self.img_txt_enc.img_enc.parameters())
        params += list(self.img_txt_enc.img_proj.parameters())
        params += list(self.img_txt_enc.cap_proj.parameters())
        params += list(self.img_txt_enc.transformer_encoder_1.parameters())

        params += list(self.img_txt_enc.image_aggregation.parameters())
        params += list(self.img_txt_enc.text_aggregation.parameters())

        if not self.opt.shared_transformer:
            params += list(self.img_txt_enc.transformer_encoder_2.parameters())

        ret.append(params)

        ret.append(list(self.img_txt_enc.txt_enc.parameters()))

        return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb, img_emb_set, cap_emb_seq, img_lengths, cap_lengths):
        """Compute the loss given pairs of image and caption embeddings
        """
        # bs = img_emb.shape[0]
        losses = {}

        img_emb_set = img_emb_set.permute(1, 0, 2)
        cap_emb_seq = cap_emb_seq.permute(1, 0, 2)
        alignment_loss = self.alignment_criterion(img_emb_set, cap_emb_seq, img_lengths, cap_lengths)
        losses.update({'alignment-loss': alignment_loss})
        self.logger.update('alignment_loss', alignment_loss.item(), img_emb_set.size(0))

        # self.logger.update('Le', matching_loss.item() + alignment_loss.item(), img_emb.size(0) if img_emb is not None else img_emb_set.size(1))
        return losses

    def forward(self, images, targets, img_lengths, cap_lengths, boxes=None, ids=None, *args):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = self.img_txt_enc.txt_enc.word_embeddings(
                captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_lengths = self.forward_emb(images, text, img_lengths,
                                                                                         cap_lengths, boxes)
        # NOTE: img_feats and cap_feats are S x B x dim

        loss_dict = self.forward_loss(img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_lengths, cap_lengths)
        return loss_dict
