import argparse
import sys
import os
import logging
import numpy as np

import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam

from sdim import SDIM
from imagenet_loader import Loader
from utils import cal_parameters, AverageMeter


class LinearResBlock(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.block_nonlinear = nn.Sequential(nn.Linear(in_size, in_size),
                                                   nn.BatchNorm1d(in_size),
                                                   nn.ReLU(),
                                                   nn.Linear(in_size, out_size))

        self.block_shortcut = nn.Linear(in_size, out_size)
        self.block_ln = nn.LayerNorm(out_size)

    def forward(self, x):
        return self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))


class ResnetWrapper(torch.nn.Module):
    def __init__(self, model: nn.Module, in_size, out_size):
        super().__init__()

        layers = list(model.children())
        self.conv_layers = torch.nn.Sequential(*layers[:-2])
        self.conv_layers.requires_grad_(requires_grad=False)
        self.avg_pool = layers[-2]
        self.lin_res = LinearResBlock(in_size, out_size)

    def forward(self, x):
        """
        Forward and extract the last conv layer output and final output.
        :param x:
        :return:
        """
        conv_out = self.conv_layers(x)

        out = self.avg_pool(conv_out).squeeze(dim=-1).squeeze(dim=-1)
        out = self.lin_res(out)

        return conv_out, out


def get_model(model_name='resnet18'):
    if model_name == 'resnet18':
        m = ResnetWrapper(models.resnet18(pretrained=True))
    elif model_name == 'resnet34':
        m = ResnetWrapper(models.resnext34(pretrained=True))
    elif model_name == 'resnet50':
        m = ResnetWrapper(models.resnext50(pretrained=True))
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
    classifier = get_model(model_name=hps.base_classifier).to(hps.device)
    logger.info('Base classifier name: {}, # parameters: {}'.format(hps.base_classifier, cal_parameters(classifier)))

    local_channel = hps.get(hps.base_classifier).last_conv_channel
    sdim = SDIM(disc_classifier=classifier,
                mi_units=hps.mi_units,
                n_classes=hps.n_classes,
                margin=hps.margin,
                local_channel=local_channel).to(hps.device)

    # logging the SDIM desc.
    for desc in sdim.desc():
        logger.info(desc)

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
            logger.info('Train Acc@1: {:.3f}, Acc@5: {:.3f}'.format(top1.avg, top5.avg))

            if losses.avg < loss_optimal:
                loss_optimal = losses.avg
                model_path = 'SDIM_{}.pth'.format(hps.base_classifier)

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
