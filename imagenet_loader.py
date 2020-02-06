import os
import cv2
import torch

from tensorpack.dataflow import LMDBSerializer, LocallyShuffleData, BatchData, MultiProcessMapDataZMQ, imgaug


"""
====== DataFlow =======
"""

def fbresnet_augmentor(train=True):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_CUBIC
    # linear seems to have more stable performance.
    # but we keep cubic for compatibility with old models
    if train:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.

            # imgaug.RandomOrderAug(
            #     [imgaug.BrightnessScale((0.6, 1.4), clip=False),
            #      imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
            #      imgaug.Saturation(0.4, rgb=False),
            #      # rgb-bgr conversion for the constants copied from fb.resnet.torch
            #      imgaug.Lighting(0.1,
            #                      eigval=np.asarray(
            #                          [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
            #                      eigvec=np.array(
            #                          [[-0.5675, 0.7192, 0.4009],
            #                           [-0.5808, -0.0045, -0.8140],
            #                           [-0.5836, -0.6948, 0.4203]],
            #                          dtype='float32')[::-1, ::-1]
            #                      )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


class Loader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        # cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
        #     to the GPU for you (necessary because this lets us to uint8 conversion on the
        #     GPU, which is faster).
    """

    def __init__(self, mode, batch_size=256, shuffle=False, num_workers=25, cache=50000, device='cuda'):
        # enumerate standard imagenet augmentors
        imagenet_augmentors = fbresnet_augmentor(mode == 'train')

        # load the lmdb if we can find it
        base_dir = '/userhome/cs/u3003679/'
        lmdb_loc = os.path.join(base_dir, 'ILSVRC-{}.lmdb'.format(mode))
        #lmdb_loc = os.path.join(os.environ['IMAGENET'], 'ILSVRC-%s.lmdb'%mode)
        ds = LMDBSerializer.load(lmdb_loc, shuffle=shuffle)
        ds = LocallyShuffleData(ds, cache)

        # ds = td.LMDBDataPoint(ds)

        def f(dp):
            x, label = dp
            x = cv2.imdecode(x, cv2.IMREAD_COLOR)
            for aug in imagenet_augmentors:
                x = aug.augment(x)
            return x, label

        ds = MultiProcessMapDataZMQ(ds, num_proc=8, map_func=f)
        # ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        # ds = AugmentImageComponent(ds, imagenet_augmentors)

        # ds = td.PrefetchData(ds, 5000, 1)

        # ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        # ds = td.AugmentImageComponent(ds, imagenet_augmentors)
        # ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = BatchData(ds, batch_size)
        # self.ds = MultiProcessRunnerZMQ(self.ds, 4)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def __iter__(self):
        for x, y in self.ds:
            x = torch.ByteTensor(x).to(self.device)
            y = torch.IntTensor(y).to(self.device)
            # but once they're on the gpu, we'll need them in
            yield uint8_to_float(x), y.long()

    def __len__(self):
        return self.ds.size()


def uint8_to_float(x):
    x = x.permute(0, 3, 1, 2) # pytorch is (n,c,w,h)
    return x.float() / 128. - 1.


if __name__ == '__main__':
    from tqdm import tqdm
    os.environ['IMAGENET'] = '/home/xwang/data/imagenet'
    dl = Loader('train', cuda=True)
    for x, y in tqdm(dl, total=len(dl)):
        # print(x.size(), y.size())
        # break
        pass
