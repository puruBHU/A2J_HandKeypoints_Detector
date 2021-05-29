# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:37:12 2021

@author: Purnendu Mishra
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from CustomDataloaderOneHand import OneHandDataloader


root = Path(r"E:\Dataset\onehand10k")
test_file = Path.cwd() / "onehand10k_test_data.csv"

BATCH_SIZE = None
IMG_SHAPE = 224

loader = OneHandDataloader(
    root=root,
    datafile=test_file,
    batch_size=BATCH_SIZE,
    img_shape=IMG_SHAPE,
    augment=True,
    normalize=True,
    crop=True,
)


ds = loader.dataset_loader(mode="test")
#%%
img, label = next(iter(ds))
label = label.numpy()
label[label < 0] = 0
x = label[:, 0]
y = label[:, 1]

x *= 224
y *= 224

plt.imshow(img.numpy().astype(np.float32))
plt.scatter(x, y, c="r")
plt.show()
