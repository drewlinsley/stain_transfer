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
from skimage.filters import gaussian
from src.pl_modules.network_tools import get_network 
import matplotlib.patches as mpatches


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



data = np.load("/media/data_cifs/projects/prj_connectomics/celltypes_v0_test.npz")
images = data["images"]

image = images[:, :-1] / 255.
labels = images[:, -1].astype(int)
imn = -17
imn = 10

patch_size = 400
stride_size = 340
diff = patch_size - stride_size
half_diff = diff // 2

# image = image.transpose((2, 0, 1))

ckpt = "/home/dlinsley/experiments/2023-01-17/16-31-06/experiments/2023-01-17/16-31-06/u19_pilot/uhkyicsi/checkpoints/epoch=105-step=1999.ckpt"
ckpt = "/home/dlinsley/experiments/2023-01-16/22-26-25/experiments/2023-01-16/22-26-25/u19_pilot/22ca4rxo/checkpoints/epoch=2622-step=49836.ckpt"
ckpt = "/home/dlinsley/experiments/2023-01-18/18-58-15/experiments/2023-01-18/18-58-15/u19_pilot/vew7mt8r/checkpoints/epoch=375-step=3759.ckpt"
# ckpt = "/home/dlinsley/experiments/2023-01-16/21-46-41/experiments/2023-01-16/21-46-41/u19_pilot/1wnx3erj/checkpoints/epoch=80-step=1538.ckpt"
# ckpt = "/home/dlinsley/experiments/2023-01-16/18-55-10/experiments/2023-01-16/18-55-10/u19_pilot/1wekpuci/checkpoints/epoch=250-step=4768.ckpt"

net = get_network("resunet_restaining_celltype_input")
net = weights_update(model=net, checkpoint=torch.load(ckpt))
net = net.to("cuda")

net.eval()
# image = image[..., :1000, :1000]
with torch.no_grad():
    output = net(torch.tensor(image).float().cuda()[[imn]])

# plt.figure()
# plt.subplot(121);plt.imshow(labels.squeeze()[imn]);plt.subplot(122);plt.imshow(np.argmax(gaussian(output[1].squeeze(0).cpu().numpy(), sigma=2., preserve_range=True, channel_axis=0), 0));plt.show()
# plt.subplot(121);plt.imshow(labels.squeeze()[imn]);plt.subplot(122);plt.imshow(np.argmax(gaussian(output[1].squeeze(0).cpu().numpy(), sigma=2., preserve_range=True, channel_axis=0), 0) * (image[imn, 1] > 0.09).astype(np.float32));plt.show()

from skimage.measure import label
from scipy import stats


segs = label(labels.squeeze()[imn] > 0)

labs = np.unique(segs)
labs = labs[labs > 0]

canvas = np.zeros_like(image[imn, 0])
pred = np.argmax(output[1].squeeze(0).cpu(), 0) * (image[imn, 1] > 0.09).astype(np.float32)
for l in labs:
    mask = l == segs
    g = pred[mask]
    g = g[g > 0]
    if len(g):
        v = stats.mode(g)[0][0]
        canvas[mask] = v  # Modal voting
        pred[mask] = 0
# import pdb;pdb.set_trace()
# plt.imshow(canvas);plt.show()

# pred += canvas  # Add the smoothed preds back
pred = canvas

plt.subplot(131)
plt.axis("off")
plt.imshow((np.concatenate((image[imn, :2].transpose((1, 2, 0)), image[imn, 1][..., None]), -1) * 255).astype(np.uint8))
# plt.imshow(np.concatenate((image[imn, :2].transpose((1, 2, 0)), image[imn, 1][..., None]), -1))
plt.title("PolyT and DAPI")
# plt.imshow(image[imn, 1])
# plt.subplot(143)
# plt.imshow(np.argmax(output[1].squeeze(0).cpu(), 0))
plt.subplot(132)
plt.title("Predicted cell types")
plt.axis("off")
plt.imshow(pred, cmap="inferno")
plt.subplot(133)
plt.imshow(labels.squeeze()[imn], cmap="inferno")
plt.axis("off")
plt.title("Partial ground truth")

# red_patch = mpatches.Patch(color='red', label='Excitatory')
# blue_patch = mpatches.Patch(color='blue', label='Microglia')
# plt.legend(handles=[red_patch, blue_patch])

plt.show()

