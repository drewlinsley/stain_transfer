import numpy as np
from glob2 import glob
import os
from skimage import io
from tqdm import tqdm
import pickle


# drwxrwxrwx 2 cis-storage cis-storage      98304 Jan 13 00:51  norm_polyt_dapi
# drwxrwxrwx 2 cis-storage cis-storage      98304 Jan 13 00:53  celltype_label

image_path = "/media/data_cifs/projects/prj_connectomics/norm_polyt_dapi"

all_images = np.asarray(glob(os.path.join(image_path, "*")))
labels = [x.replace("norm_polyt_dapi", "celltype_label") for x in all_images]

ims = [io.imread(x).astype(np.float32) for x in all_images]
ims = [255. * (x / x.max((1, 2), keepdims=True)) for x in ims]  # Normalize to uint8 range
ims = np.asarray(ims)

# Lets concat labels+images together and slice when training
labs = [io.imread(x).astype(np.float32) for x in labels]
labs = np.asarray(labs)[:, None]
extra_lab = 0
n_samples = labs.size
classes, counts = np.unique(labs, return_counts=True)
n_classes = len(classes)
weights = {cl: n_samples / (n_classes * co) for cl, co in zip(classes, counts)}
print(weights)
np.save("celltype_weights", weights)
print("Max label: {}".format(labs.max()))
print("Weights: {}".format(weights))

ims = np.concatenate((ims, labs), 1)

exp_name = "celltypes_v0"
test_split = 0.1

np.random.seed(123)

# Lets split at patients
split_idx = np.arange(len(ims))
# split_idx = np.arange(int(len(ims) * test_split))  # np.random.permutation(len(ims))
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

