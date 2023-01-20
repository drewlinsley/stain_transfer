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


from skimage.measure import label
from scipy import stats


net.eval()
# image = image[..., :1000, :1000]
correct, guess, gt = [], [], []
with torch.no_grad():
    for imn in np.arange(len(images)):
        output = net(torch.tensor(image).float().cuda()[[imn]])
        segs = label(labels.squeeze()[imn] > 0)

        labs = np.unique(segs)
        labs = labs[labs > 0]

        pred = np.argmax(output[1].squeeze(0).cpu(), 0) * (image[imn, 1] > 0.07).astype(np.float32)
        for l in labs:
            mask = l == segs
            g = pred[mask]
            g = g[g > 0]
            l = stats.mode(labels.squeeze()[imn][mask])[0][0]
            if len(g):
                v = stats.mode(g)[0][0]
                correct.append(float(v == l))
                guess.append(v)
            else:
                correct.append(0.)
                guess.append(0.)
            gt.append(l)
acc = np.mean(correct)
guess = np.asarray(guess)
gt = np.asarray(gt)

# Permutation testing
ts = []
its = 1000
for i in range(its):
    ts.append(np.mean(guess == gt[np.random.permutation(len(gt))]))
ts = np.asarray(ts)
p_value = (np.mean(acc < ts) + 1.) / float(its + 1.)

print("Accuracy: {}, p: {}".format(acc, p_value))

