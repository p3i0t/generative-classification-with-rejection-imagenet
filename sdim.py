import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from losses.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss
from mi_networks import MI1x1ConvNet

from utils import cal_parameters


class ClassConditionalGaussianMixture(nn.Module):
    """
    Gaussian (diagonal) embedding layer for class conditional distributions.
    """
    def __init__(self, n_classes, embed_size):
        super().__init__()
        self.n_classes = n_classes
        self.embed_size = embed_size
        self.class_embed = nn.Embedding(n_classes, embed_size * 2)

    def log_lik(self, x, mean, log_sigma):
        tmp = math.log(2 * math.pi) + 2 * log_sigma + (x - mean).pow(2) * torch.exp(-2 * log_sigma)
        ll = -0.5 * tmp
        return ll

    def forward(self, x):
        """
        Evaluate the log likelihood of class conditionals.
        :param x: input tensor.
        :return: log likelihoods of x.
        """
        # create all class labels for each sample x
        y_full = torch.arange(self.n_classes).unsqueeze(dim=0).repeat(x.size(0), 1).view(-1).to(x.device)

        # repeat each sample for n_classes times
        x = x.unsqueeze(dim=1).repeat(1, self.n_classes, 1).view(x.size(0) * self.n_classes, -1)

        mean, log_sigma = torch.split(self.class_embed(y_full), split_size_or_sections=self.embed_size, dim=-1)

        # evaluate log-likelihoods for each possible (x, y) pairs
        ll = self.log_lik(x, mean, log_sigma).sum(dim=-1).view(-1, self.n_classes)
        return ll


def compute_dim_loss(l_enc, m_enc, measure, mode):
    '''Computes DIM loss.
    Args:
        l_enc: Local feature map encoding.
        m_enc: Multiple globals feature map encoding.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(l_enc, m_enc, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(l_enc, m_enc)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(l_enc, m_enc)
    else:
        raise NotImplementedError(mode)

    return loss


# class FeatureTransformer(nn.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.f = nn.Sequential(nn.Linear(in_size, 2 * out_size),
#                                nn.BatchNorm1d(2 * out_size),
#                                nn.ReLU(),
#                                # nn.Linear(2 * in_size, 2 * in_size),
#                                # nn.BatchNorm1d(2 * in_size),
#                                # nn.ReLU(),
#                                nn.Linear(2 * out_size, out_size))
#
#     def forward(self, x):
#         return self.f(x)


class SDIM(torch.nn.Module):
    def __init__(self, disc_classifier, n_classes=1000, mi_units=512, margin=5., local_channel=512):
        super().__init__()
        self.disc_classifier = disc_classifier
        self.disc_classifier.requires_grad_(requires_grad=False)  # shut down grad on pre-trained classifier.

        self.n_classes = n_classes
        self.local_channel = local_channel
        self.mi_units = mi_units
        self.margin = margin

        # 1x1 conv performed on only channel dimension
        self.local_MInet = MI1x1ConvNet(self.local_channel, self.mi_units)
        self.global_MInet = MI1x1ConvNet(self.n_classes, self.mi_units)

        self.class_conditional = ClassConditionalGaussianMixture(self.n_classes, self.n_classes)

    def desc(self):
        """
        Description of this model.
        :return: tuple of descriptions of SDIM components.
        """
        n_fixed = cal_parameters(self.disc_classifier, filter_func=lambda x: not x.requires_grad)
        n_trainable = cal_parameters(self.disc_classifier, filter_func=lambda x: x.requires_grad)
        n_T = cal_parameters(self.local_MInet) + cal_parameters(self.global_MInet)
        n_C = cal_parameters(self.class_conditional)

        base_desc = 'Base classifier, # fixed parameters: {}, # trainable parameters: {}'.format(n_fixed, n_trainable)
        T_desc = 'MI evaluation network, #parameters: {}.'.format(n_T)
        class_con_desc = 'Class conditional embedding layer, #parameters: {}.'.format(n_C)
        return base_desc, T_desc, class_con_desc

    def _T(self, L, G):
        # All globals are reshaped as 1x1 feature maps.
        global_size = G.size()[1:]
        if len(global_size) == 1:
            G = G[:, :, None, None]

        L = self.local_MInet(L)
        G = self.global_MInet(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)
        return L, G

    def forward(self, x, y, measure='JSD', mode='fd'):
        """
        Evaluate the losses in SDIM.
        :param x:
        :param y:
        :param measure: measure of MI lower bound.
        :param mode:
        :return: losses.
        """
        with torch.no_grad():
            local_features, rep = self.disc_classifier(x)

        # rep = self.feature_transformer(logits)

        # compute mutual infomation loss
        L, G = self._T(local_features, rep)
        mi_loss = compute_dim_loss(L, G, measure, mode)

        # evaluate log-likelihoods as logits
        log_lik = self.class_conditional(rep)
        ll = log_lik / self.rep_size

        # mask of positive class conditional likelihood
        pos_mask = torch.zeros(rep.size(0), self.n_classes).to(rep.device).scatter(1, y.unsqueeze(dim=1), 1.)

        # compute nll loss
        nll_loss = -(ll * pos_mask).sum(dim=1).mean()

        pos_ll = torch.masked_select(ll, pos_mask.bool())
        neg_ll = torch.masked_select(ll, (1 - pos_mask).bool())
        assert pos_ll.size(0) == x.size(0)
        gap_ll = pos_ll.unsqueeze(dim=1) - neg_ll

        # log-likelihood margin loss
        ll_margin = F.relu(self.margin - gap_ll).mean()

        # total loss
        loss = mi_loss + nll_loss + ll_margin
        return loss, mi_loss, nll_loss, ll_margin, log_lik

    def infer(self, x):
        """
        Evaluate the class (Gaussian) conditionals as logits of generative classifier.
        :param x: input tensor.
        :return: logits.
        """
        with torch.no_grad():
            local_features, rep = self.disc_classifier(x)
        # rep = self.feature_transformer(logits)
        log_lik = self.class_conditional(rep)
        return log_lik


