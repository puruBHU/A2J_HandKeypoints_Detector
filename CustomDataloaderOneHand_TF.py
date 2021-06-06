# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:24:13 2021

@author: Purnendu Mishra
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from CustomDataloaderOneHand_OpenCV import OneHandDataloader as LOADER


class OneHandDataloader(LOADER):
    def __init__(self, *args, **kwargs):
        super(OneHandDataloader, self).__init__(*args, **kwargs)

    def image_reader(self, img_size=(224, 224)):
        def f(filePath, labels):
            channels = 3
            img_string = tf.io.read_file(filePath)
            img_decoded = tf.image.decode_jpeg(img_string, channels=channels)

            if self.crop:
                boxes, labels = tf.numpy_function(
                    self.pyfunc_encode_labels, [labels], [tf.float32, tf.float32]
                )

                boxes = tf.reshape(boxes, (1, -1))

                box_indices = tf.constant((0,), dtype=tf.int32)
                crop_size = [img_size[0], img_size[1]]
                img = tf.expand_dims(img_decoded, axis=0)
                img = tf.image.crop_and_resize(img, boxes, box_indices, crop_size)
                img = tf.squeeze(img, axis=0)

            else:
                img = tf.image.resize(
                    img_decoded, [img_size[0], img_size[1]], method="bilinear"
                )
            return tf.cast(img, dtype=tf.uint8), labels

        return f

    def rescale_labels(self, keypoints=None, bndbox=None):
        y_top, x_top, y_bot, x_bot = bndbox

        keypoints = keypoints.reshape(-1, 2)

        non_zero = np.nonzero(keypoints[:, 0])[0]

        bndbox_w = x_bot - x_top
        bndbox_h = y_bot - y_top

        keypoints[non_zero, 0] -= x_top
        keypoints[non_zero, 1] -= y_top

        keypoints[non_zero, 0] /= bndbox_w
        keypoints[non_zero, 1] /= bndbox_h

        return keypoints

    def get_bbox(self, keypoints=None):

        keypoints = keypoints.reshape(-1, 2)  # shape: (21,2)
        keypoints[keypoints < 0] = 0

        x_values = keypoints[:, 0]
        y_values = keypoints[:, 1]

        # Non Zero Values X and Y keypoints values
        x_non_zero = x_values[np.nonzero(x_values)[0]]
        y_non_zero = y_values[np.nonzero(y_values)[0]]

        # # These are offset values for extracted bounding box coordinates
        nx = 0.05
        ny = 0.05

        xtop = max(min(x_non_zero) - nx, 0)
        ytop = max(min(y_non_zero) - ny, 0)

        xbot = min(max(x_non_zero) + nx, 0.99)
        ybot = min(max(y_non_zero) + ny, 0.99)

        return (ytop, xtop, ybot, xbot)


#%%
if __name__ == "__main__":
    from pathlib import Path
    from matplotlib import pyplot as plt
    from skimage import io

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
        normalize=True,
        crop=True,
        horizontal_flip=True,
        vertical_flip=True,
        color_jitter=True,
    )

    ds = loader.dataset_loader()
    #%%
    img, label = next(iter(ds))
    img = img.numpy().astype(np.float32)
    label = label.numpy()
    label[label < 0] = 0
    x = label[:, 0]
    y = label[:, 1]

    x *= 224
    y *= 224

    # io.imshow(img)

    plt.imshow(img.astype(np.float32))
    plt.scatter(x, y, c="r")
    plt.show()
