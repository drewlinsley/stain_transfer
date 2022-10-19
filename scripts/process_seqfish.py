import numpy as np
from glob2 import glob
import os
from skimage import io
from tqdm import tqdm
import pickle


image_path = "/media/data_cifs/projects/prj_connectomics/seqfish/*.npy"
all_images = np.asarray(glob(os.path.join(image_path)))
ims = [np.load(x).astype(np.float32) for x in all_images]

ims = [255. * (x / x.max((1, 2), keepdims=True)) for x in ims]  # Normalize to uint8 range
ims = np.asarray(ims)

exp_name = "seqfish_v0"
test_split = 0.1

np.random.seed(123)
split_idx = np.random.permutation(len(ims))
test_idx = split_idx[:int(len(ims) * test_split)]
train_idx = split_idx[int(len(ims) * test_split):]

test_ims = ims[test_idx]
train_ims = ims[train_idx]
unique_image_types = [0]  # What are our categories?
# labels

# NOTE: Do not currently have labels for individual images (i.e., patient status, cogdx, etc)
# Using dummy data for now
train_data = {
    "images": train_ims,
    "labels": [0],
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
np.savez(os.path.join("/media/data_cifs/projects/prj_connectomics/", "{}_train".format(exp_name)), **train_data)
test_data = {
    "images": test_ims,
    "labels": [0],  # proc_labels_test,
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
np.savez(os.path.join("/media/data_cifs/projects/prj_connectomics/", "{}_test".format(exp_name)), **test_data)

