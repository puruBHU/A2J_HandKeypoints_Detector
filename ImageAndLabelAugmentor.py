# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:52:42 2021

@author: Purnendu Mishra
"""

import numpy as np
import tensorflow as tf


def random_horizontal_flip(image, label):
    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    if cond:
        image, label = flip_image_and_label_left_right(image, label)

    return image, label


def random_vertical_flip(image, label):
    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    if cond:
        image, label = flip_image_and_label_up_down(image, label)

    return image, label


def flip_image_and_label_left_right(image, label):
    def flip_label(label):
        label = label.reshape(-1, 2)
        label[label < 0.0] = 0
        non_zero_idx = np.nonzero(label[:, 0])
        label[non_zero_idx, 0] = 1.0 - label[non_zero_idx, 0]

        label[label == 0.0] = -1.0
        return label

    image = tf.image.flip_left_right(image)
    label = tf.numpy_function(flip_label, [label], tf.float32)

    return image, label


def flip_image_and_label_up_down(image, label):
    def flip_label(label):
        label = label.reshape(-1, 2)
        label[label < 0.0] = 0
        non_zero_idx = np.nonzero(label[:, 0])
        label[non_zero_idx, 1] = 1.0 - label[non_zero_idx, 1]

        label[label == 0.0] = -1.0
        return label

    image = tf.image.flip_up_down(image)
    label = tf.numpy_function(flip_label, [label], tf.float32)

    return image, label


def randomly_adjust_brightness(image, label):

    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    image = tf.cond(
        cond, lambda: tf.image.random_brightness(image, 0.25), lambda: image
    )
    return image, label


def randomly_adjust_contrast(image, label):

    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    image = tf.cond(
        cond, lambda: tf.image.random_contrast(image, 0.25, 1.75), lambda: image
    )
    return image, label


def randomly_adjust_saturation(image, label):

    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    image = tf.cond(
        cond, lambda: tf.image.random_saturation(image, 0.75, 1.25), lambda: image
    )
    return image, label


def randomly_adjust_hue(image, label):

    uniform_random = tf.random.uniform([], 0, 1.0)
    cond = tf.less(uniform_random, 0.5)
    image = tf.cond(cond, lambda: tf.image.random_hue(image, 0.04), lambda: image)
    return image, label
