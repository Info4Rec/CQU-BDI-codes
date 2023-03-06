import numpy as np
import torch
from torch import nn as nn
from torchvision import models as models
from .utils import l2norm


def EncoderImage(opt):

    embed_size = opt.embed_size
    finetune = opt.image_finetune
    cnn_type = opt.image_model_type
    img_enc = EncoderImageFull(embed_size, finetune, cnn_type)

    return img_enc


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False, avgpool_size=(4, 4)):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            raise NotImplementedError
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.spatial_feats_dim = self.cnn.module.fc.in_features
            modules = list(self.cnn.module.children())[:-2]
            self.cnn = torch.nn.Sequential(*modules)
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
            self.glob_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.spatial_feats_dim, embed_size)

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model.cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        spatial_features = self.cnn(images)
        features = self.glob_avgpool(spatial_features)   # compute a single feature
        spatial_features = self.avgpool(spatial_features)   # fix the size of the spatial grid


        features = torch.flatten(features, 1)
        # normalization in the image embedding space
        features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(features)

        features = l2norm(features)

        return features

    def get_finetuning_params(self):
        return list(self.cnn.parameters())

