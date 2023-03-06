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
    opt.checkpoint = '/home/1718/yl/yl_code/1_running_code/MNTE/digital/runs/f30k-relation/model_best_rsum.pth.tar'
    opt.size = '1k'   # 5k

    main(opt)
