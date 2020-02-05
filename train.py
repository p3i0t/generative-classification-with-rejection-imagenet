import argparse
import sys
import os
import numpy as np
from tqdm import tqdm

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


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--n_classes", type=int,
                        default=1000, help="number of classes of dataset.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=256, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=200, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")

    # SDIM hyperparams:
    parser.add_argument("--mi_units", type=int,
                        default=512, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=512, help="size of the global representation from encoder")
    parser.add_argument("--margin", type=float,
                        default=3., help="likelihood margin.")
    parser.add_argument("--base_classifier", type=str, default='resnext50_32x4d',
                        help="name of base discriminative classifier.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    # Models
    print('Base classifier name: {}'.format(hps.classifier_name))
    classifier = get_model(model_name=hps.classifier_name).to(hps.device)

    sdim = SDIM(disc_classifier=classifier,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                n_classes=hps.n_classes,
                margin=hps.margin).to(hps.device)

    train_loader = Loader('train', device=hps.device)
    test_loader = Loader('test', device=hps.device)

    if use_cuda and hps.n_gpu > 1:
        sdim = torch.nn.DataParallel(sdim, device_ids=list(range(hps.n_gpu)))

    optimizer = Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()), lr=hps.lr)

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)


    def train_epoch():
        """
        One epoch training.
        """
        sdim.train()

        losses = AverageMeter('Loss')
        MIs = AverageMeter('MI')
        nlls = AverageMeter('NLL')
        margins = AverageMeter('Margin')
        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')

        for x, y in tqdm(train_loader, total=len(train_loader)):
            # backward
            optimizer.zero_grad()

            if use_cuda and hps.n_gpu > 1:
                f_forward = sdim.module.eval_losses
            else:
                f_forward = sdim.eval_losses

            loss, mi_loss, nll_loss, ll_margin, log_lik = f_forward(x, y)
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))

            MIs.update(mi_loss.item(), x.size(0))
            nlls.update(nll_loss.item(), x.size(0))
            margins.update(ll_margin.item(), x.size(0))

        print('Train loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
            losses.avg, MIs.avg, nlls.avg, margins.avg
        ))
        print('Train Acc@1: {:.3f}, Acc@5: {:.3f}'.format(top1.avg, top5.avg))

        return losses.avg, top1.avg, top5.avg

    def inference():
        """
        One epoch testing.
        """
        sdim.eval()

        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(hps.device)
                y = y.to(hps.device)

                # eval logits
                log_lik = sdim(x)
                acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
                top1.update(acc1, x.size(0))
                top5.update(acc5, x.size(0))

        print('Test Acc@1: {:.3f}, Acc@5: {:.3f}'.format(top1.avg, top5.avg))
        return top1.avg, top5.avg

    loss_optimal = 1e5
    for epoch in range(1, hps.epochs + 1):
        print('===> Epoch: {}'.format(epoch))
        train_loss, train_acc_top1, train_acc_top5 = train_epoch()
        test_acc_top1, test_acc_top5 = inference()

        if train_loss < loss_optimal:
            loss_optimal = train_loss
            model_path = 'SDIM_{}_MI{}_rep{}.pth'.format(hps.classifier_name, hps.mi_units, hps.rep_size)

            if use_cuda and hps.n_gpu > 1:
                state = sdim.module.state_dict()
            else:
                state = sdim.state_dict()

            check_point = {'model_state': state,
                           'train_acc_top1': train_acc_top1,
                           'train_acc_top5': train_acc_top5,
                           'test_acc_top1': test_acc_top1,
                           'test_acc_top5': test_acc_top5}

            torch.save(state, os.path.join(hps.log_dir, model_path))