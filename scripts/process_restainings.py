import numpy as np
from glob2 import glob
import os
from skimage import io
from tqdm import tqdm
import pickle


image_path = "/media/data_cifs/projects/prj_connectomics/*.npy"
# all_images = np.asarray(glob(os.path.join(image_path)))
all_images = [
        "/media/data_cifs/projects/prj_connectomics/he.npy",
        "/media/data_cifs/projects/prj_connectomics/polyt.npy"
]
ims = [np.load(x).astype(np.float32) for x in all_images]
ims = [255. * (x / x.max()) for x in ims]

proc_images_test = proc_images_train = ims
proc_labels_train = [proc_images_train[0][2000:], proc_images_train[1][2000]]
proc_images_test = [proc_images_test[0][:2000], proc_images_test[1][:2000]]
unique_image_types = [0]

train_data = {
    "images": proc_images_train,
    "labels": proc_labels_train,
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
np.savez(os.path.join("/media/data_cifs/projects/prj_connectomics/", "restaining_train"), **train_data)
test_data = {
    "images": proc_images_test,
    "labels": [0],  # proc_labels_test,
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
np.savez(os.path.join("/media/data_cifs/projects/prj_connectomics/", "restaining_test"), **test_data)

