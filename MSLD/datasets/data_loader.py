from torch.utils.data import DataLoader
from .collate_batch import collate_fn
from .dataset import BaseDataSet
#from .dataset_label import BaseDataSet
from .sampler import RandomIdentitySampler
from .transforms import build_transforms


def build_data(cfg, is_train=0):
    if is_train == 1:
        transforms = build_transforms(cfg, is_train)
        dataset = BaseDataSet(cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        sampler = RandomIdentitySampler(dataset=dataset,
                                        batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                                        num_instances=cfg.DATA.NUM_INSTANCES,
                                        max_iters=cfg.SOLVER.MAX_ITERS
                                       )
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 batch_sampler=sampler,
                                 sampler = None,
                                 num_workers=cfg.DATA.NUM_WORKERS,
                                 pin_memory=False
                                 )
    else :
        transforms = build_transforms(cfg, is_train)
        dataset = BaseDataSet(cfg.DATA.TEST_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE)
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 batch_size=cfg.DATA.TEST_BATCHSIZE,
                                 num_workers=cfg.DATA.NUM_WORKERS
                                 )
    return data_loader
