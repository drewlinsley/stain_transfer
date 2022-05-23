from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10 as cifar10_data
from torch.nn import functional as F
import numpy as np
from glob2 import glob
from skimage import io
import pickle


DATADIR = "data/"


def load_image(directory):
    return Image.open(directory).convert('L')


def invert(img):

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def colour(img, ch=0, num_ch=3):

    colimg = [torch.zeros_like(img)] * num_ch
    # colimg[ch] = img
    # Use beta distribution to push the mixture to ch 1 or ch 2
    if ch == 0:
        rand = torch.distributions.beta.Beta(0.5, 1.)
    elif ch == 1:
        rand = torch.distributions.beta.Beta(1., 0.5) 
    else:
        raise NotImplementedError("Only 2 channel images supported now.")
    rand = rand.sample()
    colimg[0] = img * rand
    colimg[1] = img * (1 - rand)
    return torch.cat(colimg)


class CIFAR10(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform

        self.dataset = cifar10_data(root=DATADIR, download=True)
        self.data_len = len(self.dataset)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        img = np.asarray(img)
        label = np.asarray(label)
        # Transpose shape from H,W,C to C,H,W
        img = img.transpose(2, 0, 1).astype(np.float32)
        # img = F.to_tensor(img)
        # label = F.to_tensor(label)
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class u19_pilot(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.crop_size = [224, 224]

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        # self.data = np.load(self.path)
        with open(self.path, 'rb') as handle:
            self.data = pickle.load(handle)

        self.images = self.data["images"]
        self.labels = self.data["labels"]
        self.image_channels = self.data["image_channels"]
        self.morph_channel = np.where(self.image_channels == "polyT_histology")[0][0]

        del self.data

        self.images = [x.astype(np.float32) for x in self.images]
        self.morphology_images, self.channel_images = [], []
        for x in self.images:
            mimage = x[self.morph_channel] / 255.
            # mimage = torch.from_numpy(mimage).cuda()
            self.morphology_images.append(mimage)  # x[self.morph_channel] / 255.)
            keeps = np.arange(len(x)).tolist()
            keeps.pop(self.morph_channel)
            cimage = x[keeps]
            # cimage = torch.from_numpy(cimage).cuda()

            # Binarize cimage
            cimage = (cimage > (255. / 5)).astype(cimage.dtype)
            self.channel_images.append(cimage)  # x[keeps])

        batch_size = 30
        num_gpus = 5
        steps = 10
        self.data_len = batch_size * num_gpus * steps  # 12 * 4 * 10  # 0  # len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        # Grab a random patient
        numsubs = len(self.images)
        sub = np.random.randint(numsubs)
        sel_img = self.morphology_images[sub]
        sel_channels = self.channel_images[sub]
        class_label = self.labels[sub]
        imshape = sel_img.shape

        # Now a random crop
        h = np.random.randint(low=0, high=imshape[0] - self.crop_size[0])
        w = np.random.randint(low=0, high=imshape[1] - self.crop_size[1])
        # h = torch.randint(low=0, high=imshape[0] - self.crop_size[0], size=[], device=sel_img.device)
        # w = torch.randint(low=0, high=imshape[1] - self.crop_size[1], size=[], device=sel_img.device)
        # imgs = sel_img[:, h: h + self.crop_size[0], w: w + self.crop_size[1]]
        # morphology_img = imgs[self.morph_channel] / 255.  # Normalize morphology
        morphology_img = sel_img[h: h + self.crop_size[0], w: w + self.crop_size[1]]
        morphology_img = morphology_img[None]
        channel_label = sel_channels[:, h: h + self.crop_size[0], w: w + self.crop_size[1]]
        # keeps = np.arange(len(imgs)).tolist()
        # keeps.pop(self.morph_channel)
        # channel_label = imgs[keeps]
        return morphology_img, class_label, channel_label, sub

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class restainings(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.crop_size = [400, 400]  # [224, 224]

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        # self.data = np.load(self.path)

        self.data = np.load(self.path)  # , allow_pickle=True)

        self.images = self.data["images"]
        self.labels = [0]

        del self.data.f
        self.data.close
        del self.data

        self.images = [x.astype(np.float32) for x in self.images]
        self.morphology_images, self.channel_images = [], []

        mimage = self.images[0] / 255.
        self.morphology_images.append(mimage)
        self.channel_images.append(self.images[1][None])
        batch_size = 30
        num_gpus = 5
        steps = 10
        self.data_len = batch_size * num_gpus * steps  # 12 * 4 * 10  # 0  # len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        # Grab a random patient
        numsubs = len(self.morphology_images)
        sub = 0  # np.random.randint(numsubs)
        sel_img = self.morphology_images[sub]
        sel_channels = self.channel_images[sub]
        class_label = self.labels[sub]
        imshape = sel_img.shape

        # Now a random crop
        h = np.random.randint(low=0, high=imshape[0] - self.crop_size[0])
        w = np.random.randint(low=0, high=imshape[1] - self.crop_size[1])
        # h = torch.randint(low=0, high=imshape[0] - self.crop_size[0], size=[], device=sel_img.device)
        # w = torch.randint(low=0, high=imshape[1] - self.crop_size[1], size=[], device=sel_img.device)
        # imgs = sel_img[:, h: h + self.crop_size[0], w: w + self.crop_size[1]]
        # morphology_img = imgs[self.morph_channel] / 255.  # Normalize morphology
        morphology_img = sel_img[h: h + self.crop_size[0], w: w + self.crop_size[1]]
        morphology_img = morphology_img[None]
        channel_label = sel_channels[:, h: h + self.crop_size[0], w: w + self.crop_size[1]]
        channel_label = channel_label.repeat(3, axis=0)
        # keeps = np.arange(len(imgs)).tolist()
        # keeps.pop(self.morph_channel)
        # channel_label = imgs[keeps]
        return morphology_img, class_label, channel_label, sub

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class NuclearGedi(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        curated = True if ".npz" in path else False
        if curated:
            data = np.load(path)
            files = data["files"]
            labels = data["labels"]
            # files = np.asarray(data["files"]).reshape(-1, 1)
            # labels = np.asarray(data["labels"]).reshape(-1, 1)
            # files = torch.from_numpy(files)
            labels = torch.from_numpy(labels)
        else:
            # List all the files
            print("Globbing files for NuclearGedi, this may take a while...")
            live = glob(os.path.join(self.path, "GC150nls-Live", "*.tif"))
            dead = glob(os.path.join(self.path, "GC150nls-Dead", "*.tif"))
            if not len(live) or not len(dead):
                raise RuntimeError("No files found at {}".format(self.path))
            files = np.asarray(live + dead)
            labels = np.concatenate((np.ones(len(live)), np.zeros(len(dead))), 0).astype(int).reshape(-1, 1)
        np.random.seed(42)
        idx = np.random.permutation(len(files))
        files = files[idx]  # Shuffle
        labels = labels[idx]
        self.files = files
        self.labels = labels
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        label = self.labels[index]
        label = label.reshape(-1)
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class NonNuclearGedi(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        curated = True if ".npz" in path else False
        if curated:
            data = np.load(path)
            files = data["files"]
            labels = data["labels"]
            # files = np.asarray(data["files"]).reshape(-1, 1)
            # labels = np.asarray(data["labels"]).reshape(-1, 1)
            # files = torch.from_numpy(files)
            labels = torch.from_numpy(labels)
        else:
            # List all the files
            print("Globbing files for NuclearGedi, this may take a while...")
            live = glob(os.path.join(self.path, "GC150nls-Live", "*.tif"))
            dead = glob(os.path.join(self.path, "GC150nls-Dead", "*.tif"))
            if not len(live) or not len(dead):
                raise RuntimeError("No files found at {}".format(self.path))
            files = np.asarray(live + dead)
            labels = np.concatenate((np.ones(len(live)), np.zeros(len(dead))), 0).astype(int).reshape(-1, 1)
        np.random.seed(42)
        idx = np.random.permutation(len(files))
        files = files[idx]  # Shuffle
        labels = labels[idx]
        self.files = files
        self.labels = labels
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        label = self.labels[index]
        label = label.reshape(-1)
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"

