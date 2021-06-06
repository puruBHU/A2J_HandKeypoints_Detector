# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 22:52:40 2021

@author: Purnendu Mishra
"""


import cv2
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from CustomDataloaderOneHand_OpenCV import OneHandDataloader as LOADER


class OneHandDataloader(LOADER):
    def __init__(self, *args, **kwargs):
        super(OneHandDataloader, self).__init__(*args, **kwargs)

    def pyfunc_read_image(self, imgPath=None, target_size=None):
        imgPath = imgPath.decode("utf-8")
        with Image.open(imgPath) as img:
            img = img.convert("RGB")
            img = np.array(img)

        img = cv2.resize(
            img, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC
        )

        return tf.cast(img, dtype=tf.uint8)

    def pyfunc_read_and_crop_image(self, imgPath=None, bbox=None, target_size=None):

        imgPath = imgPath.decode("utf-8")
        with Image.open(imgPath) as img:
            img = img.convert("RGB")
            img = np.array(img)

        h, w = img.shape[:2]

        if bbox is not None:
            xtop, ytop, xbot, ybot = bbox

            xtop = int(w * xtop)
            ytop = int(h * ytop)
            xbot = int(w * xbot)
            ybot = int(h * ybot)

            img = img[ytop:ybot, xtop:xbot, :]

        else:
            raise ValueError("The bounding box coordiates are not provided")

        img = cv2.resize(
            img, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC
        )

        return tf.cast(img, dtype=tf.uint8)


#%%

if __name__ == "__main__":
    from pathlib import Path
    from matplotlib import pyplot as plt

    root = Path(r"E:\Dataset\onehand10k")
    test_file = Path.cwd() / "onehand10k_train_data.csv"

    BATCH_SIZE = None
    IMG_SHAPE = 224

    loader = OneHandDataloader(
        root=root,
        mode="train",
        datafile=test_file,
        batch_size=BATCH_SIZE,
        img_shape=IMG_SHAPE,
        normalize=False,
        crop=True,
        horizontal_flip=True,
        vertical_flip=True,
        color_jitter=True,
    )

    ds = loader.dataset_loader()
    #%%
    img, label = next(iter(ds))
    label = label.numpy()
    label[label < 0] = 0
    x = label[:, 0]
    y = label[:, 1]

    x *= 224
    y *= 224

    plt.imshow(img.numpy())
    plt.scatter(x, y, c="r")
    plt.show()

    # def pyfunc_image_reader(self, filename):
    #     filename = filename.decode("utf-8")
    #     with Image.open(filename) as img:
    #         img = img.convert("RGB")
    #         img = np.array(img)
    #         img = tf.convert_to_tensor(img, dtype=tf.uint8)

    #     return img
