import numpy as np
from glob2 import glob
import os
from skimage import io
from tqdm import tqdm
import pickle


image_path = "/media/data_cifs/projects/prj_connectomics/images"
all_images = np.asarray(glob(os.path.join(image_path, "**", "*.tif")))

start = 1
pad = 22
crop = 1000
min_hist = .8  # Must have 80% nonzero pixels

image_types = [x.split(os.path.sep)[-2] for x in all_images]
patients, labels = [],[]
for x in all_images:
    if "_ad_" in x:
        label = 1
    else:
        label = 0
    labels.append(label)

    fs = x.split(os.path.sep)[-1]
    patients.append(fs.split("_")[1])
labels = np.asarray(labels)
patients = np.asarray(patients)

unique_image_types = np.unique(image_types)
unique_patients = np.unique(patients)

images = {}
for idx, it in enumerate(unique_image_types):
    if it not in images:
        images[it] = idx

# Load images then package them up into 4-channel tensors
proc_images, proc_labels = [], []
# for patient, label in zip(patients, labels):
for patient in tqdm(unique_patients, total=len(unique_patients)):

    # Find all_images that are for this patient
    patient_images = []
    for image in all_images:
        if patient in image:
            patient_images.append(image)
            if "_ad_" in image:
                label = 1
            else:
                label = 0
    patient_images = np.asarray(patient_images)

    # Load images in a fixed order
    loaded_images = np.asarray([io.imread(x) for x in patient_images]).astype(np.float32)
    loaded_idx = []
    for idx, im in enumerate(patient_images):
        im = im.split(os.path.sep)[-2]
        loaded_idx.append(np.where(im == unique_image_types)[0])
    loaded_idx = np.asarray(loaded_idx).ravel()
    loaded_images = loaded_images[loaded_idx]

    # Normalize here
    loaded_images = loaded_images / np.asarray([1, 1, 1000, 1000])[:, None, None]
    loaded_images = (loaded_images * 255.).astype(np.uint8)
    proc_images.append(loaded_images)
    proc_labels.append(label)

proc_labels = np.asarray(proc_labels)
proc_images_test = [proc_images[3]]
proc_labels_test = proc_labels[[3]]
proc_images_train = proc_images
proc_images_train.pop(3)
proc_labels_train = np.delete(proc_labels, 3)

import pdb;pdb.set_trace()
train_data = {
    "images": proc_images_train,
    "labels": proc_labels_train,
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
with open(os.path.join("/media/data_cifs/projects/prj_connectomics/", "u19_data_train.pkl"), 'wb') as f:
    pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
test_data = {
    "images": proc_images_test,
    "labels": proc_labels_test,
    "image_channels": unique_image_types,
    "label_key": ["control", "ad"]
}
with open(os.path.join("/media/data_cifs/projects/prj_connectomics/", "u19_data_test.pkl"), 'wb') as f:
    pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# np.savez(
#     os.path.join("/media/data_cifs/projects/prj_connectomics/", "u19_data_train"),
#     images=proc_images_train,
#     labels=proc_labels_train,
#     image_channels=unique_image_types,
#     label_key=["control", "ad"])

# np.savez(
#     os.path.join("/media/data_cifs/projects/prj_connectomics/", "u19_data_test"),
#     images=proc_images_test,
#     labels=proc_labels_test,
#     image_channels=unique_image_types,
#     label_key=["control", "ad"])
