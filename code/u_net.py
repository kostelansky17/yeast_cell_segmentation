from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Input,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dropout,
    MaxPool2D,
)

from settings import CONV_LAYER_SETTINGS, SELECTED_U_NET_SETTINGS


def build_u_net(input_size, filters, u_depth):
    """
    Builds U-shaped neural net (U-net) as proposed by Ronneberger, Fischer, Brox
    in U-Net: Convolutional Networks for Biomedical Image Segmentation.

    Parameters
    ----------
    input_size : int
        Size of the squared input image
    filters : int
        Number of filters to be used
    u_depth : int
        Depth of the U-net

    Returns
    -------
    model : tensorflow.keras.Sequential
        U-net model
    """
    input_layer = Input(shape=(input_size, input_size, 1), name="input_layer")

    residual_connections = []
    for i in range(u_depth):
        if i == 0:
            x = Conv2D(filters, **CONV_LAYER_SETTINGS)(input_layer)
        else:
            x = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)

        x = Dropout(0.1)(x)
        residual = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)
        residual_connections.append(residual)
        x = MaxPool2D(pool_size=(2, 2))(residual)
        filters *= 2

    padding = [184, 88, 40, 16, 4]
    for i in range(u_depth):
        x = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)
        filters = int(filters / 2)
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2))(x)
        x = concatenate([Cropping2D(padding.pop())(residual_connections.pop()), x])

    x = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters, **CONV_LAYER_SETTINGS)(x)
    output_layer = Conv2D(1, (1, 1), 1, activation=sigmoid)(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def main(u_net_settings):
    """
    FOR TESTING PURPOSES ONLY

    Builds an U-net model and prints its summary

    Parameters
    ----------
    u_net_settings : tuple
        Settings used to build the model
    """
    model = build_u_net(*u_net_settings)
    print(model.summary())


if __name__ == "__main__":
    main(SELECTED_U_NET_SETTINGS["model_settings"])
