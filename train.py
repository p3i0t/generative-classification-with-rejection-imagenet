import argparse
import sys
import os
import logging
import numpy as np

import hydra
from omegaconf import DictConfig

import torch
import torchvision.models as models
from torch.optim import Adam

from sdim import SDIM
from imagenet_loader import Loader
from utils import cal_parameters, AverageMeter


def get_model(model_name='resnet18'):
    if model_name == 'resnet18':
        m = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        m = models.resnext34(pretrained=True)
    elif model_name == 'resnet50':
        m = models.resnext50(pretrained=True)
    print('Model name: {}, # parameters: {}'.format(model_name, cal_parameters(m)))
    return m


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum() / batch_size * 100
        res.append(correct_k.item())
    return res


@hydra.main(config_path='config.yaml')
def train(hps: DictConfig) -> None:
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    logger = logging.getLogger(__name__)

    cuda_available = torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    device = "cuda" if cuda_available and hps.device == 'cuda' else "cpu"

    # Models
    print('Base classifier name: {}'.format(hps.base_classifier))
    classifier = get_model(model_name=hps.base_classifier).to(hps.device)

    sdim = SDIM(disc_classifier=classifier,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                n_classes=hps.n_classes,
                margin=hps.margin).to(hps.device)

    train_loader = Loader('train', batch_size=hps.n_batch_train, device=device)

    if cuda_available and hps.n_gpu > 1:
        sdim = torch.nn.DataParallel(sdim, device_ids=list(range(hps.n_gpu)))

    optimizer = Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()), lr=hps.lr)

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    loss_optimal = 1e5
    n_iters = 0

    losses = AverageMeter('Loss')
    MIs = AverageMeter('MI')
    nlls = AverageMeter('NLL')
    margins = AverageMeter('Margin')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for x, y in train_loader:
        n_iters += 1
        if n_iters == hps.training_iters:
            break

        # backward
        optimizer.zero_grad()
        loss, mi_loss, nll_loss, ll_margin, log_lik = sdim(x, y)
        loss.mean().backward()
        optimizer.step()

        acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))

        MIs.update(mi_loss.item(), x.size(0))
        nlls.update(nll_loss.item(), x.size(0))
        margins.update(ll_margin.item(), x.size(0))

        if n_iters % hps.log_interval == hps.log_interval - 1:
            logger.info('Train loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
                losses.avg, MIs.avg, nlls.avg, margins.avg
            ))

            if losses.avg < loss_optimal:
                loss_optimal = losses.avg
                model_path = 'SDIM_{}_MI{}_rep{}.pth'.format(hps.base_classifier, hps.mi_units, hps.rep_size)

                if cuda_available and hps.n_gpu > 1:
                    state = sdim.module.state_dict()
                else:
                    state = sdim.state_dict()

                check_point = {'model_state': state,
                               'train_acc_top1': top1.avg,
                               'train_acc_top5': top5.avg}

                torch.save(check_point, os.path.join(hps.log_dir, model_path))

            losses.reset()
            MIs.reset()
            nlls.reset()
            margins.reset()
            top1.reset()
            top5.reset()


if __name__ == '__main__':
    train()
