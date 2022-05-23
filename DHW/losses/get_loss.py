
from .triplet_loss import TripletLoss

import torch



def get_loss(method):
    losses = {
        'Triplet': TripletLoss(margin=0.5)
    }
    return losses[method]



