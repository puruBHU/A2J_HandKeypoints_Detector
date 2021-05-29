# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:24:13 2021

@author: Purnendu Mishra
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from data_augmentor import *


class OneHandDataloader(object):
    def __init__(
        self,
        root=None,
        batch_size=1,
        img_shape=224,
        channels=3,
        datafile=None,
        augment=True,
        normalize=True,
        shuffle=True,
        crop=False,
    ):
        self.root = root
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.datafile = datafile
        self.channels = channels
        self.augment = augment
        self.normalize = normalize
        self.shuffle = shuffle
        self.crop = crop

    def dataset_loader(self, mode="train"):
        img_source_path = self.root / mode.capitalize() / "source"

        df = pd.read_csv(self.datafile, header=None)
        df[0] = df[0].map(lambda x: str(img_source_path.joinpath(x)))

        data = df.to_numpy()

        img_absolute_path = np.array(data[:, 0], dtype="str")
        labels = np.array(data[:, 1:], dtype=np.float32)
        labels = self.process_labels(labels)

        dataset = tf.data.Dataset.from_tensor_slices((img_absolute_path, labels))
        dataset = self.pipeline(
            ds=dataset,
            batch_size=self.batch_size,
            img_shape=(self.img_shape,) * 2,
            channels=self.channels,
            crop=self.crop,
        )
        return dataset

    def pipeline(
        self, ds=None, batch_size=None, img_shape=(224, 224), channels=3, crop=False
    ):
        """

        Args:
            ds (tf.data, optional): DESCRIPTION. Defaults to None.
            batch_size (int, optional): DESCRIPTION. Defaults to None.

        Returns:
            ds (tf.data): DESCRIPTION.

        """
        AUTOTUNE = tf.data.AUTOTUNE

        img_reader = image_reader(img_shape=img_shape, channels=channels, crop=crop)
        ds = ds.map(img_reader, num_parallel_calls=AUTOTUNE)

        if self.augment:
            ds = ds.map(data_augmentation, num_parallel_calls=AUTOTUNE)

        if self.normalize:
            ds = ds.map(normalize_image, num_parallel_calls=AUTOTUNE)

        if self.shuffle:
            ds = ds.shuffle(1000)

        if batch_size is not None:
            ds = ds.batch(batch_size)

        ds = ds.prefetch(AUTOTUNE)
        return ds

    def process_labels(self, label):
        """

        Args:
            label (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
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


def data_augmentation(img, label):
    if tf.random.uniform((1,), 0, 1) > 0.5:
        img, label = horizontal_flip(img, label)

    if tf.random.uniform((1,), 0, 1) > 0.5:
        img, label = vertical_flip(img, label)

    # if tf.random.uniform((1,), 0, 1) > 0.5:
    #     img = adjust_gamma(img)

    # img = color_augmentation(img)

    return img, label


def normalize_image(imgFile, labels):
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


def image_reader(img_shape=(224, 224), channels=3, crop=False):
    def f(filePath, labels):
        img_string = tf.io.read_file(filePath)
        img_decoded = tf.image.decode_jpeg(img_string, channels=channels)

        if crop:
            boxes = tf.numpy_function(get_bbox, [labels], tf.float32)
            boxes = tf.reshape(boxes, (1, -1))

            box_indices = tf.constant((0,), dtype=tf.int32)
            crop_size = [img_shape[0], img_shape[1]]
            img = tf.expand_dims(img_decoded, axis=0)
            img = tf.image.crop_and_resize(img, boxes, box_indices, crop_size)
            img = tf.squeeze(img, axis=0)
            labels = tf.numpy_function(label_encoder, [labels, boxes], tf.float32)
        else:
            img = tf.image.resize(
                img_decoded, [img_shape[0], img_shape[1]], method="bilinear"
            )
        return img, labels

    return f


def get_bbox(keypoints=None):

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

    bbox_width = xbot - xtop
    bbox_height = ybot - ytop

    return tf.cast((ytop, xtop, ybot, xbot), tf.float32)


def label_encoder(label=None, boxes=None):
    boxes = tf.squeeze(boxes, axis=0)
    y_top, x_top, y_bot, x_bot = boxes
    label = label.reshape(-1, 2)
    label[label < 0] = 0

    non_zero_idx = np.nonzero(label[:, 0])[0]

    box_width = x_bot - x_top
    box_height = y_bot - y_top

    label[non_zero_idx, 0] -= x_top
    label[non_zero_idx, 1] -= y_top

    label[non_zero_idx, 0] /= box_width
    label[non_zero_idx, 1] /= box_height

    label[label == 0] = -1

    return label
