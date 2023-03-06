import evaluation
import torch
from config import DefaultConfig

def main(opt):
    model_checkpoint = opt.checkpoint

    checkpoint = torch.load(model_checkpoint)
    print('Checkpoint loaded from {}'.format(model_checkpoint))

    evaluation.evalrank(opt, checkpoint, split="test")


if __name__ == '__main__':
    opt = DefaultConfig()
    opt.checkpoint = './runs/f30k/global/model_best_rsum.pth.tar'
    opt.size = '1k'

    main(opt)
