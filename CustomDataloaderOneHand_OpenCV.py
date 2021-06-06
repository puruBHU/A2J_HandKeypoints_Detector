# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:24:13 2021

@author: Purnendu Mishra
"""
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

from ImageAndLabelAugmentor import *


class OneHandDataloader(object):
    def __init__(
        self,
        root=None,
        mode=None,
        batch_size=1,
        img_shape=224,
        datafile=None,
        normalize=True,
        shuffle=True,
        crop=False,
        color_jitter=False,
        horizontal_flip=False,
        vertical_flip=False,
    ):
        self.root = root
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.datafile = datafile
        self.normalize = normalize
        self.shuffle = shuffle
        self.crop = crop
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.mode = mode

        self.data = self.data_initializer()

    def data_initializer(self):
        img_source_path = self.root / self.mode.capitalize() / "source"

        df = pd.read_csv(self.datafile, header=None)
        df[0] = df[0].map(lambda x: str(img_source_path.joinpath(x)))

        data = df.to_numpy()

        img_absolute_path = list(data[:, 0])
        labels = np.array(data[:, 1:], dtype=np.float32)
        labels = self.process_labels(labels)

        return (img_absolute_path, labels)

    def dataset_loader(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.data)
        dataset = self.pipeline(
            ds=dataset,
            batch_size=self.batch_size,
            img_shape=(self.img_shape,) * 2,
            crop=self.crop,
        )
        return dataset

    def image_reader(self, img_size=(None, None)):
        def f(imgPath, labels):

            if self.crop:
                bbox, labels = tf.numpy_function(
                    self.pyfunc_encode_labels, [labels], [tf.float32, tf.float32]
                )

                image = tf.numpy_function(
                    self.pyfunc_read_and_crop_image, [imgPath, bbox, img_size], tf.uint8
                )

            else:
                image = tf.numpy_function(
                    self.pyfunc_read_image, [imgPath, img_size], tf.uint8
                )

            return image, labels

        return f

    def pyfunc_read_image(self, imgPath=None, target_size=None):
        imgPath = imgPath.decode("utf-8")
        img = cv2.imread(imgPath)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img = cv2.resize(
            img, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC
        )

        return tf.cast(img, dtype=tf.uint8)

    def pyfunc_read_and_crop_image(self, imgPath=None, bbox=None, target_size=None):
        # imgPath = str(imgPath.numpy())
        imgPath = imgPath.decode("utf-8")
        img = cv2.imread(imgPath)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if bbox is not None:
            xtop, ytop, xbot, ybot = bbox

            xtop = int(w * xtop)
            ytop = int(h * ytop)
            xbot = int(w * xbot)
            ybot = int(h * ybot)

            img = img[ytop:ybot, xtop:xbot, :]

        img = cv2.resize(
            img, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC
        )

        return tf.cast(img, dtype=tf.uint8)

    def pyfunc_encode_labels(self, labels):
        bbox = self.get_bbox(labels)
        labels = self.rescale_labels(labels, bbox)

        return (tf.cast(bbox, dtype=tf.float32), tf.cast(labels, dtype=tf.float32))

    def pipeline(self, ds=None, batch_size=None, img_shape=(224, 224), crop=False):

        AUTOTUNE = tf.data.AUTOTUNE

        img_reader = self.image_reader(img_size=img_shape)

        ds = ds.map(img_reader, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)

        # Random Color Jitter
        if self.color_jitter:
            ds = ds.map(self.random_color_jitter, num_parallel_calls=AUTOTUNE)

        if self.horizontal_flip:
            ds = ds.map(random_horizontal_flip, num_parallel_calls=AUTOTUNE)

        if self.vertical_flip:
            ds = ds.map(random_vertical_flip, num_parallel_calls=AUTOTUNE)

        if self.normalize:
            ds = ds.map(self.normalize_image, num_parallel_calls=AUTOTUNE)

        if self.shuffle:
            ds = ds.shuffle(1000)

        if self.batch_size is not None:
            ds = ds.batch(self.batch_size)

        return ds

    def random_color_jitter(self, img, label):
        img, label = randomly_adjust_contrast(img, label)
        img, label = randomly_adjust_hue(img, label)
        img, label = randomly_adjust_brightness(img, label)
        img, label = randomly_adjust_saturation(img, label)
        return img, label

    def normalize_image(self, imgFile, labels):
        """

        Args:
            imgFile (TYPE): DESCRIPTION.
            labels (TYPE): DESCRIPTION.

        Returns:
            img (TYPE): DESCRIPTION.
            labels (TYPE): DESCRIPTION.

        """
        img = tf.cast(imgFile, tf.float32)
        img = tf.truediv(img, 255.0)

        return img, labels

    def process_labels(self, label):

        N = len(label)
        keypoints = []

        for n in range(N):
            lb = label[n]

            width = lb[0]
            height = lb[1]
            keys = lb[3:].reshape(-1, 2)

            # Normalize the x-coordinates
            keys[:, 0] = keys[:, 0] / width

            # Normalize the y-coordinates
            keys[:, 1] = keys[:, 1] / height

            keys[keys < 0] = -1
            keypoints.append(keys)

        return np.array(keypoints, dtype=np.float32)

    def rescale_labels(self, keypoints=None, bndbox=None):
        x_top, y_top, x_bot, y_bot = bndbox

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

        return xtop, ytop, xbot, ybot


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
    # x = label[:, 0]
    # y = label[:, 1]

    # x *= 224
    # y *= 224

    # plt.imshow(img.numpy())
    # plt.scatter(x, y, c="r")
    # plt.show()
