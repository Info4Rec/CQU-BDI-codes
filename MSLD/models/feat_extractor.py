import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import pdb
import math
import argparse
import sys

def feat_extractor(model, data_loader, cfg, logger):
    print("models extract test feature.........")
    print("dataloader:",len(data_loader))
    model.eval()
    feats = list()

    for i, batch in enumerate(data_loader):
        imgs = batch[0] 
        batch_size = len(imgs)
        with torch.no_grad():
            imgs = imgs.cuda()
            feature, feature_local = model.googlelayer(imgs)
            outputs = model.embedding_layer(feature)
            outputs = (outputs).data.cpu().numpy()
            feats.append(outputs)
     
        if (i + 1) % 100 == 0:
            logger.info(f'Extract Features: [{i + 1}/{len(data_loader)}]')
        del outputs
        del feature, feature_local
 
    feats = np.vstack(feats)
    return feats
