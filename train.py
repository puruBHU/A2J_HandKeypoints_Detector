# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:48:14 2020

@author: Purnendu Mishra
"""

# Standard library import
import argparse
from pathlib import Path

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

# Local library import
from CustomDataloaderOneHand import OneHandDataloader
from Model import A2J

# %%
parser = argparse.ArgumentParser()

parser.add_argument(
		"-b", "--batch_size", default = 1, type = int, help = "Batch size to be used for training"
		)

parser.add_argument(
		"-e",
		"--epochs",
		default = 100,
		type = int,
		help = "Defines the number of epochs the model will be trained",
		)

args = parser.parse_args()

# %%
TARGET_SIZE = 176

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

# %%

root = Path(r"E:\Dataset\onehand10k")
train_file = Path.cwd() / "onehand10k_train_data.csv"
val_file = Path.cwd() / "onehand10k_test_data.csv"

train_dataset = OneHandDataloader(
		root = root,
		datafile = train_file,
		batch_size = BATCH_SIZE,
		img_shape = TARGET_SIZE,
		normalize = True,
		shuffle = True,
		augment = True,
		)

val_dataset = OneHandDataloader(
		root = root,
		datafile = val_file,
		batch_size = BATCH_SIZE,
		img_shape = TARGET_SIZE,
		normalize = True,
		)

train_loader = train_dataset.dataset_loader(mode = "train")
val_loader = val_dataset.dataset_loader(mode = "test")

# %%
huber = Huber(delta = 1.0)

model = A2J(input_shape = TARGET_SIZE + (3,), keys = 21)
opt = Adam(lr = 0.00035, beta_1 = 0.5)
# # opt      = SGD(lr = initial_lr, momentum=0.9, nesterov=True, decay=1e-6)

model.compile(optimizer = opt, loss = huber, metrics = ["mae"])

# %%
save_path = Path.cwd() / "checkpoints"

if not save_path.exists():
	save_path.mkdir(parents = True)

checkpoints = ModelCheckpoint(
		"{}/A2J_OneHand_Cropped_".format(save_path) + "{epoch:03d}_{val_loss:0.4f}.hdf5",
		monitor = "val_loss",
		mode = "auto",
		verbose = 1,
		save_best_only = True,
		save_freq = "epoch",
		)

records = CSVLogger("logs/A2J_OneHand_Cropped_training.log")

callbacks = [checkpoints, records]
# %%
print("Started training the model ...")
model.fit(
		x = train_loader,
		validation_data = val_loader,
		epochs = EPOCHS,
		steps_per_epoch = len(train_loader),
		callbacks = callbacks,
		use_multiprocessing = False,
		workers = 4,
		verbose = 2,
		initial_epoch = 0,
		)

# model.save("A2J_OneHand__Cropped_Final_weights.hdf5")
