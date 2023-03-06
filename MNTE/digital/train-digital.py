import os
import time
import shutil
import numpy as np

import torch
import pytorch_warmup as warmup

import data
from models.loss import AlignmentContrastiveLoss
from utils import get_model
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn

import logging
from torch.utils.tensorboard import SummaryWriter

from config import DefaultConfig

def main(**Kwargs):
    # Hyper Parameters
    opt = DefaultConfig()
    opt.parse(Kwargs)

    # Initialize logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt)

    # Construct the model
    model = get_model(opt)
    if torch.cuda.is_available() and not opt.resume:
        model.cuda()

    # Get the paramaters of model
    params, secondary_lr_multip = model.get_parameters()
    all_params = params[0] + params[1]
    if len(all_params) != len(list(model.parameters())):
        raise ValueError('Not all parameters are being returned! Correct get_parameters() method')

    if secondary_lr_multip > 0:
        optimizer = torch.optim.Adam([{'params': params[0]},
                                      {'params': params[1], 'lr': opt.lr * secondary_lr_multip}],
                                     lr=opt.lr)
    else:
        optimizer = torch.optim.Adam(params[0], lr=opt.lr)

    # LR scheduler
    scheduler_name = opt.scheduler
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=opt.gamma)
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))

    # Warmup scheduler
    warmup_scheduler_name = opt.warmup if not opt.resume else None
    if warmup_scheduler_name == 'linear':
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=opt.warmup_period)
    elif warmup_scheduler_name is None:
        warmup_scheduler = None
    else:
        raise ValueError('{} warmup scheduler is not available'.format(warmup_scheduler_name))

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        filename = opt.resume
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()
            if opt.resume:
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['scheduler'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                # Eiters
                model.Eiters = checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.resume, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # Train the Model
    best_rsum = 0

    for epoch in range(start_epoch, opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader,
              measure=opt.measure, grad_clip=opt.grad_clip,
              scheduler=scheduler, warmup_scheduler=warmup_scheduler)

        # evaluate on validation set
        rsum = validate(val_loader, model, tb_logger, measure=opt.measure, log_step=opt.log_step)

        # remember best R@ sum and save checkpoint
        is_best_rsum = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)


        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best_rsum, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader, measure='cosine', grad_clip=-1, scheduler=None, warmup_scheduler=None):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train()
        if scheduler is not None:
            scheduler.step(epoch)

        if warmup_scheduler is not None:
            warmup_scheduler.dampen()

        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss_dict = model(*train_data)
        loss = sum(loss for loss in loss_dict.values())

        # compute gradient and do SGD step
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.add_scalar('epoch', epoch, model.Eiters)
        tb_logger.add_scalar('step', i, model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val, model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val, model.Eiters)
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(val_loader, model, tb_logger, measure=measure, log_step=opt.log_step)


def validate(val_loader, model, tb_logger, measure='cosine', log_step=10):
    # Encode the data
    img_embs, cap_embs, img_lengths, cap_lengths = encode_data(
        model, val_loader, log_step, logging.info)

    img_embs = img_embs.cpu().numpy()
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    cap_embs = cap_embs.cpu().numpy()

    # Initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(return_similarity_mat=True)

    # Calculate the similarity matrix
    sims = shard_xattn(sim_matrix_fn, img_embs, cap_embs, img_lengths, cap_lengths)

    # Image to text
    (r1, r5, r10), rsum = i2t(sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, rsum))

    # Text to image
    (r1i, r5i, r10i), risum = t2i(sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, risum))

    # The sum of recall@
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # Record metrics in tensorboard
    tb_logger.add_scalar('r1', r1, model.Eiters)
    tb_logger.add_scalar('r5', r5, model.Eiters)
    tb_logger.add_scalar('r10', r10, model.Eiters)
    tb_logger.add_scalar('r1i', r1i, model.Eiters)
    tb_logger.add_scalar('r5i', r5i, model.Eiters)
    tb_logger.add_scalar('r10i', r10i, model.Eiters)
    tb_logger.add_scalar('rsum', currscore, model.Eiters)

    return currscore


def save_checkpoint(state, is_best_rsum, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best_rsum:
        shutil.copyfile(prefix + filename, prefix + 'model_best_rsum.pth.tar')


if __name__ == '__main__':
    main()
