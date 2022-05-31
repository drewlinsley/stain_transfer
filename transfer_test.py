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
fn = "9158123_h&e_stitched_real"
# fn = "4781406_h&e_stitched"
# image = "/media/data_cifs/projects/prj_connectomics/4781406_h&e_stitched.tif"
mode = "he_polyt"

# fn = "4781406_oligoDT_DLPFC_1_polyT_stitched"
# mode = "polyt_he"

image = "/media/data_cifs/projects/prj_connectomics/{}.tif".format(fn)
image = io.imread(image).astype(np.float32)
image = image / image.max()

net = get_network("resunet_restaining")
if mode == "he_polyt":
    net = weights_update(model=net, checkpoint=torch.load("experiments/2022-05-27/09-14-39/experiments/2022-05-27/09-14-39/restainings/b7s1nh76/checkpoints/epoch=1367-step=194256.ckpt"))
else:
    net = weights_update(model=net, checkpoint=torch.load("experiments/2022-05-27/09-22-46/experiments/2022-05-27/09-22-46/restainings/2nusbcyu/checkpoints/epoch=1329-step=188860.ckpt"))
net = net.to("cuda")

perfs = []
net.eval()

h_range = np.arange(0, image.shape[0], stride_size)
w_range = np.arange(0, image.shape[1], stride_size)

transformed_image = np.zeros_like(image).astype(np.float32)
for h in tqdm(h_range, desc="Processing", total=len(h_range)):  #  tqdm(range(1000), desc="Processing", total=1000):
    for w in w_range:
        x = image[h: h + patch_size, w: w + patch_size]
        if x.shape[0] < patch_size or x.shape[1] < patch_size:
            continue
        x = torch.from_numpy(x).to("cuda")
        x = x[None, None]
        ad_pred, output_0, output_1, output_2 = net(x)  # sx)
        output_0 = torch.argmax(output_0.squeeze(), 0).detach().cpu().numpy()
        output_0 = output_0.astype(transformed_image.dtype)
        if h > 0:
            h_start = h + half_diff
            h_end = h_start + stride_size

            h_patch_start = half_diff
            h_patch_end = patch_size - half_diff
        else:
            h_start = h
            h_end = h_start + patch_size
            h_patch_start = 0
            h_patch_end = patch_size

        if w > 0:
            w_start = w + half_diff
            w_end = w_start + stride_size

            w_patch_start = half_diff
            w_patch_end = patch_size - half_diff
        else:
            w_start = w
            w_end = w_start + patch_size
            w_patch_start = 0
            w_patch_end = patch_size

        transformed_image[h_start: h_end, w_start: w_end] = output_0[h_patch_start: h_patch_end, w_patch_start: w_patch_end]
np.save("transformed_image_{}_{}".format(mode, fn), transformed_image)
# plt.subplot(121);plt.imshow(image[:2000, :2000], cmap="Greys");plt.axis("off");plt.subplot(122);plt.imshow(transformed_image[:2000, :2000], cmap="Greys");plt.axis("off");plt.show()

