from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataset(data_name='mnist', data_dir='data', train=True, label_id=None, crop_flip=True):
    """
    Get a dataset.
    :param data_name: str, name of dataset.
    :param data_dir: str, base directory of data.
    :param train: bool, return train set if True, or test set if False.
    :param label_id: None or int, return data with particular label_id.
    :param crop_flip: bool, whether use crop_flip as data augmentation.
    :return: pytorch dataset.
    """
    transform_1d_crop_flip = transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5,), (0.5,))  # 1-channel, scale to [-1, 1]
                                        ])

    transform_1d = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                #transforms.Normalize((0.5,), (0.5, ))
            ])

    transform_3d_crop_flip = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_3d = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if data_name == 'mnist':
        #if train:
            # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
        #    transform = transform_1d if crop_flip is False else transform_1d_crop_flip
        #else:
        #    transform = transform_1d

        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform_1d)

    elif data_name == 'fashion':
        if train:
            # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
            transform = transform_1d if crop_flip is False else transform_1d_crop_flip
        else:
            transform = transform_1d

        dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)

    elif data_name == 'cifar10':
        if train:
            # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
            transform = transform_3d if crop_flip is False else transform_3d_crop_flip
        else:
            transform = transform_3d

        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'svhn':
        if train:
            # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
            transform = transform_3d if crop_flip is False else transform_3d_crop_flip
            split = 'train'
        else:
            transform = transform_3d
            split = 'test'

        dataset = datasets.SVHN(data_dir, split=split, download=True, transform=transform)
    else:
        print('dataset {} is not available'.format(data_name))

    if label_id is not None:
        # select samples with particular label
        if data_name == 'cifar10': #isinstance(dataset.targets, list):
            # for cifar10
            targets = np.array(dataset.targets)
            idx = targets == label_id
            dataset.targets = list(targets[idx])
            dataset.data = dataset.data[idx]
        elif data_name == 'svhn':
            idx = dataset.labels == label_id
            dataset.labels = dataset.labels[idx]
            dataset.data = dataset.data[idx]
        else:
            # for MNIST and FashionMNIST
            targets = dataset.targets
            data = dataset.data
            idx = targets == label_id
            dataset.targets = targets[idx]
            dataset.data = data[idx]
    return dataset


def cal_parameters(model):
    """
    Calculate the number of parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: int, number of parameters.
    """
    cnt = 0
    for para in model.parameters():
        cnt += para.numel()
    return cnt

#
# def clean_state_dict(state_dict):
#     # see https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         assert k.startswith('module.')
#         name = k[7:]  # remove `module.`
#         new_state_dict[name] = v
#     # load params
#     return new_state_dict


if __name__ == '__main__':
    dataset = get_dataset(data_name='cifar10', train=True, label_id=1, crop_flip=False)
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)
    # dataset = get_dataset(label_id=1)
    # train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    for batch_id, (x, y) in enumerate(train_loader):
        print(x.size())
        print(y)
        break
