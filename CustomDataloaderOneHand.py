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
			root = None,
			batch_size = 1,
			img_shape = 224,
			channels = 3,
			datafile = None,
			augment = True,
			normalize = True,
			shuffle = True,
	):
		self.root = root
		self.batch_size = batch_size
		self.img_shape = img_shape
		self.datafile = datafile
		self.channels = channels
		self.augment = augment
		self.normalize = normalize
		self.shuffle = shuffle

	def dataset_loader(self, mode = "train"):
		img_source_path = self.root / mode.capitalize() / "source"

		df = pd.read_csv(self.datafile, header = None)
		df[0] = df[0].map(lambda x: str(img_source_path.joinpath(x)))

		data = df.to_numpy()

		img_absolute_path = np.array(data[:, 0], dtype = "str")
		labels = np.array(data[:, 1:], dtype = np.float32)
		labels = self.process_labels(labels)

		dataset = tf.data.Dataset.from_tensor_slices((img_absolute_path, labels))
		dataset = self.pipeline(
				ds = dataset,
				batch_size = self.batch_size,
				img_shape = (self.img_shape,) * 2,
				channels = self.channels,
		)
		return dataset

	def pipeline(self, ds = None, batch_size = None, img_shape = (224, 224), channels = 3):
		"""

		Args:
			ds (TYPE, optional): DESCRIPTION. Defaults to None.
			batch_size (TYPE, optional): DESCRIPTION. Defaults to None.

		Returns:
			ds (TYPE): DESCRIPTION.

		"""
		AUTOTUNE = tf.data.AUTOTUNE

		img_reader = image_reader(img_shape = img_shape, channels = channels)

		ds = ds.map(img_reader, num_parallel_calls = AUTOTUNE)

		if self.augment:
			ds = ds.map(data_augmentation, num_parallel_calls = AUTOTUNE)

		if self.normalize:
			ds = ds.map(normalize_image, num_parallel_calls = AUTOTUNE)

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

		return np.array(keypoints, dtype = np.float32)


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


def image_reader(img_shape = (224, 224), channels = 3):
	def f(filePath, labels):
		img_string = tf.io.read_file(filePath)
		img_decoded = tf.image.decode_jpeg(img_string, channels = channels)

		img = tf.image.resize(
				img_decoded, [img_shape[0], img_shape[1]], method = "bilinear"
		)
		return img, labels

	return f
