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
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.pl_modules.network_tools import get_network 


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint['state_dict'].items():
        k = k.replace("net.", "")
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print("Not restoring variable {}".format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


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
        with open(self.path, 'rb') as handle:
            self.data = pickle.load(handle)

        self.images = self.data["images"]
        self.labels = self.data["labels"]
        self.image_channels = self.data["image_channels"]
        self.morph_channel = 0

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
        channel_label = channel_label.repeat(3, axis=0)
        # keeps = np.arange(len(imgs)).tolist()
        # keeps.pop(self.morph_channel)
        # channel_label = imgs[keeps]
        return morphology_img, class_label, channel_label, sub

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


patch_size = 400
stride_size = 340
diff = patch_size - stride_size
half_diff = diff // 2


# -rwxrwxrwx 1 cis-storage cis-storage 402653312 Jan 10 13:06 /media/data_cifs/projects/prj_connectomics/2022_1204_ad_20223617_pos87.npy
# (selfies) dlinsley@serrep8:~/stain_transfer$ ls -lrt /media/data_cifs/projects/prj_connectomics/2022_1209_control_50400835_pos87.npy
# -rwxrwxrwx 1 cis-storage cis-storage 402653312 Jan 10 13:07 /media/data_cifs/projects/prj_connectomics/2022_1209_control_50400835_pos87.npy

image = "v2_transformed_image_polyt_he.npy"
image = "/media/data_cifs/projects/prj_connectomics/2022_1204_ad_20223617_pos87.npy"
image = "/media/data_cifs/projects/prj_connectomics/2022_1209_control_50400835_pos87.npy"
image_data = np.load(image).astype(np.float32)
image = image_data[:2]
labels = image_data[2:]
image = image / image.max((1, 2), keepdims=True)
# image = image.transpose((2, 0, 1))


ckpt = "experiments/backup-update-submission/22-08-40/experiments/2022-10-18/22-08-40/u19_pilot/peit9ut5/checkpoints/epoch=2804-step=95370.ckpt"

net = get_network("resunet_restaining_seqfish_input")
net = weights_update(model=net, checkpoint=torch.load(ckpt))
net = net.to("cuda")

net.eval()
# image = image[..., :1000, :1000]
with torch.no_grad():
    output = net(torch.tensor(image).float().cuda()[None])
import pdb;pdb.set_trace()
ch=18;plt.subplot(121);plt.imshow(output[1][0, ch, 1].cpu() > 0);plt.subplot(122);plt.imshow(labels[ch]);plt.show()
len(output)
output = output.detach().cpu()
fig = plt.figure()
plt.subplot(211)
plt.imshow(image[0])
plt.subplot(212)
plt.imshow(image[1])
plt.subbplot(213)
plt.imshow(output[0])
plt.subplot(214)
plt.imshow(output[1])
plt.show()

