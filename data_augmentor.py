# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:13:47 2021

@author: Purnendu Mishra
"""
import numpy as np
import tensorflow as tf


def horizontal_flip(img, label, seed=None):
    def flip_label(label):
        label = label.reshape(-1, 2)
        label[label < 0.0] = 0
        non_zero_idx = np.nonzero(label[:, 0])
        label[non_zero_idx, 0] = 1.0 - label[non_zero_idx, 0]

        label[label == 0.0] = -1.0
        return label

    img = tf.image.flip_left_right(img)
    label = tf.numpy_function(flip_label, [label], tf.float32)

    return img, label


def vertical_flip(img, label):
    def flip_label(label):
        label = label.reshape(-1, 2)
        label[label < 0.0] = 0
        non_zero_idx = np.nonzero(label[:, 0])
        label[non_zero_idx, 1] = 1.0 - label[non_zero_idx, 1]

        label[label == 0.0] = -1.0
        return label

    img = tf.image.flip_up_down(img)
    label = tf.numpy_function(flip_label, [label], tf.float32)

    return img, label


# def adjust_gamma(img):
#     gamma = np.random.randint(15, 300) / 100
#     img = tf.image.adjust_gamma(img, gamma=gamma)
#     img = tf.math.truediv(img, tf.math.reduce_max(img))
#     img = tf.math.multiply(img, 255)
#     return img


def color_augmentation(x):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
