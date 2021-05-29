# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:04:16 2020

@author: puru
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Softmax, Reshape, Concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import plot_model


from CustomLayer import SpatialSoftmax, KeypointsRegression

#%%


def A2J(input_shape=(None, None, 3), num_anchors=16, keys=21, mode="train"):

    filters = 128

    input_ = Input(shape=input_shape)
    net = ResNet50(include_top=False, weights="imagenet", input_tensor=input_)
    # =============================================================================
    #     change stride of level 4 in ResNet50 to 1
    # =============================================================================
    config = net.get_config()
    l = len(config["layers"])

    # level 4 starts from 143 layer in the ResNet50 configuration
    for i in range(143, l):
        if config["layers"][i]["class_name"] == "Conv2D":
            config["layers"][i]["config"]["strides"] = (1, 1)
            config["layers"][i]["config"]["dilation_rate"] = (2, 2)

    net = Model.from_config(config)
    # =============================================================================
    #    Get feature map from intermediate layers
    # =============================================================================

    # Common trunk:  CT
    f_ct = net.layers[142].output
    x = conv_relu_bn(filters=filters)(f_ct)
    x = conv_relu_bn(filters=filters)(x)
    x = conv_relu_bn(filters=filters)(x)
    x = conv_relu_bn(filters=filters)(x)

    # Anchors
    x = Conv2D(filters=num_anchors * keys, kernel_size=(3, 3), padding="same")(x)

    # A2J uses strides of 4
    strides = 4  # strides of anchors
    h = x.shape[1]
    w = x.shape[2]

    x = Reshape((h * strides, w * strides, keys))(x)

    # # Apply softmax to anchors proposal (ap)
    # anchors weights
    ap = SpatialSoftmax(name="spatial_softmax")(x)
    # =============================================================================
    #     Common trunk
    # =============================================================================
    # Regression Trunk (RT)
    f_rt = net.layers[-1].output
    y = conv_relu_bn(filters=filters)(f_rt)
    y = conv_relu_bn(filters=filters)(y)
    y = conv_relu_bn(filters=filters)(y)
    y = conv_relu_bn(filters=filters)(y)

    y = Conv2D(filters=num_anchors * keys * 2, kernel_size=(3, 3), padding="same")(y)

    # positional offsets
    os = Reshape(target_shape=(h * strides, w * strides, keys, 2))(y)

    if mode == "train":
        out = KeypointsRegression(name="keypoint_positions")([ap, os])
        out = Reshape((keys, 2))(out)
        return Model(inputs=net.input, outputs=out, name="A2J")

    elif mode == "test":
        return Model(inputs=net.input, outputs=[ap, os], name="A2J")


def conv_relu_bn(**params):

    filters = params["filters"]
    kernel_size = params.setdefault("kernel_size", (3, 3))
    dilation_rate = params.setdefault("dilation_rate", (1, 1))
    strides = params.setdefault("strides", (1, 1))
    padding = params.setdefault("padding", "same")
    kernel_init = params.setdefault("kernel_initializer", RandomNormal(stddev=0.01))
    kernel_reg = params.setdefault("kernel_regularizer", l2(0.001))

    def f(x):
        x = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            dilation_rate=dilation_rate,
        )(x)
        x = BatchNormalization(axis=-1, fused=True)(x)

        return Activation("relu")(x)

    return f


#%%

if __name__ == "__main__":
    model = A2J(input_shape=(176, 176, 3), keys=21, mode="train")
    model.summary()
    plot_model(model, to_file="A2J_Model.jpg", show_shapes=True)
