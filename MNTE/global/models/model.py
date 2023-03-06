import torch
import torch.nn.init
import torch.nn as nn
import torch.backends.cudnn as cudnn

from .loss import ContrastiveLoss
from .text import EncoderText
from .visual import EncoderImage
from .utils import l2norm


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
        embed_size = opt.embed_size
        self.img_enc = EncoderImage(opt)

        self.img_proj = nn.Linear(visual_feat_dim, embed_size)
        self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
        self.embed_size = embed_size

    def forward(self, features, captions, cap_len):
        # process captions by using bert
        full_cap_emb_aggr = self.txt_enc(captions, cap_len)  # B x S x cap_dim

        # process image regions using a two-layer transformer
        full_img_emb_aggr = self.img_enc(features)  # B x S x vis_dim

        full_cap_emb_aggr = l2norm(full_cap_emb_aggr)
        full_img_emb_aggr = l2norm(full_img_emb_aggr)


        return full_img_emb_aggr, full_cap_emb_aggr


class GlobalNet(torch.nn.Module):
    """
    The global subnetwork
    """

    def __init__(self, opt):
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(opt)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and optimizer
        self.matching_criterion = ContrastiveLoss(margin=opt.margin,
                                                  measure=opt.measure,
                                                  max_violation=opt.max_violation)

        self.Eiters = 0
        self.opt = opt

    def forward_emb(self, images, captions, cap_len):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb_aggr, cap_emb_aggr = self.img_txt_enc(images, captions, cap_len)

        return img_emb_aggr, cap_emb_aggr, cap_len

    def get_parameters(self):
        lr_multiplier = 1.0 if self.opt.text_finetune else 0.0

        ret = []
        params = list(self.img_txt_enc.img_enc.parameters())
        params += list(self.img_txt_enc.img_proj.parameters())
        params += list(self.img_txt_enc.cap_proj.parameters())

        ret.append(params)

        ret.append(list(self.img_txt_enc.txt_enc.parameters()))

        return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        losses = {}

        matching_loss = self.matching_criterion(img_emb, cap_emb)
        losses.update({'matching-loss': matching_loss})
        self.logger.update('matching_loss', matching_loss.item(), img_emb.size(0))

        return losses

    def forward(self, images, targets, cap_lengths, *args):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            text = features
        else:
            text = targets

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, cap_lengths = self.forward_emb(images, text, cap_lengths)

        loss_dict = self.forward_loss(img_emb_aggr, cap_emb_aggr)
        return loss_dict