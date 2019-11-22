from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers.core import Dropout, Reshape
from keras.layers import PReLU, Conv2DTranspose
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, \
    concatenate, Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras.losses import mean_squared_error, binary_crossentropy, sparse_categorical_crossentropy
from keras import losses
import keras.backend as K
import numpy as np
from keras.regularizers import l2
# Check Keras version - code will switch API if needed.
from keras import __version__ as keras_version
k2 = True if keras_version[0] == '2' else False

# If Keras is v2.x.x, create Keras 1-syntax wrappers.
if not k2:
    from keras.layers import merge, Input
    from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                            UpSampling2D)
else:
    from keras.layers import Concatenate, Input
    from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                            UpSampling2D)


    def merge(layers, mode=None, concat_axis=None):
        """Wrapper for Keras 2's Concatenate class (`mode` is discarded)."""
        return Concatenate(axis=concat_axis)(list(layers))


    def Convolution2D(n_filters, FL, FLredundant, activation=None,
                      init=None, W_regularizer=None, border_mode=None):
        """Wrapper for Keras 2's Conv2D class."""
        return Conv2D(n_filters, FL, activation=activation,
                      kernel_initializer=init,
                      kernel_regularizer=W_regularizer,
                      padding=border_mode)
def Conv(x, out_channels, dilation_rate=(1, 1)):
    return Conv2D(out_channels, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation_rate, padding='same')(x)
def UpConv(x, out_channels):
    return Conv2DTranspose(out_channels, kernel_size=(3, 3), strides=(2, 2), padding='same', output_padding=(1, 1))(x)
def BN_Conv_Relu(x, out_channels, dilation_rate=(1, 1)):
    x = BatchNormalization(axis=3, momentum=0.01)(x)
    x = Conv2D(out_channels, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation_rate, padding='same')(x)
    x = ReLU()(x)
    return x
def BN_UpConv_Relu(x, out_channels):
    x = BatchNormalization(axis=3, momentum=0.01)(x)
    x = UpConv(x, out_channels)
    x = Activation('relu')(x)
    return x
def ConvOut(x):
    return Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
def unet_pooling_3(dim,start_filter,lr=0.0001):
    inpt = Input(batch_shape=(None, dim, dim, 1))
    BCR3 = BN_Conv_Relu(inpt, start_filter)  # BUCR40
    BCR4 = BN_Conv_Relu(BCR3, start_filter)
    MP5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(BCR4)
    BCR6 = BN_Conv_Relu(MP5, start_filter*2)
    BCR7 = BN_Conv_Relu(BCR6, start_filter*2)  # BUCR36
    BCR8 = BN_Conv_Relu(BCR7, start_filter*2)
    MP9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(BCR8)
    BCR10 = BN_Conv_Relu(MP9, start_filter*4)
    BCR11 = BN_Conv_Relu(BCR10, start_filter*4)  # BUCR32
    BCR12 = BN_Conv_Relu(BCR11, start_filter*4)
    MP13 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(BCR12)

    BCR30 = BN_Conv_Relu(MP13, start_filter*4)
    BCR31 = BN_Conv_Relu(BCR30, start_filter*4)

    BUCR32 = BN_UpConv_Relu(BCR31, start_filter*4)  # BCR11
    Add33 = add([BUCR32, BCR11])
    BCR34 = BN_Conv_Relu(Add33, start_filter*4)
    BCR35 = BN_Conv_Relu(BCR34, start_filter*4)

    BUCR36 = BN_UpConv_Relu(BCR35, start_filter*2)  # BCR7
    Add37 = add([BUCR36, BCR7])
    BCR38 = BN_Conv_Relu(Add37, start_filter*2)
    BCR39 = BN_Conv_Relu(BCR38, start_filter*2)

    BUCR40 = BN_UpConv_Relu(BCR39, start_filter)  # BCR3
    Add41 = add([BUCR40, BCR3])
    BCR42 = BN_Conv_Relu(Add41, start_filter)
    BCR43 = BN_Conv_Relu(BCR42, start_filter)
    CO44 = ConvOut(BCR43)
    out = Conv2D(1, 1, activation='sigmoid', padding='same')(CO44)

    out = Reshape((dim, dim))(out)
    model = Model(inputs=inpt, outputs=out)  # convd2d
    optimizer = Adam(lr=lr)

    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=optimizer)
    model.summary()
    return model
#Ari Silburt's UNet for Carter decter
def unet(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network.

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter.
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2), )(a3)

    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    if k2:
        model = Model(inputs=img_input, outputs=u)
    else:
        model = Model(input=img_input, output=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',metrics=['binary_accuracy'], optimizer=optimizer)
    print(model.summary())
    return model
def unet_ConvT(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network.

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter.
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2), )(a3)

    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = Conv2DTranspose(n_filters* 2, (3, 3), strides=(2, 2), padding="same")(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    if k2:
        model = Model(inputs=img_input, outputs=u)
    else:
        model = Model(input=img_input, output=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',metrics=['binary_accuracy'], optimizer=optimizer)
    print(model.summary())
    return model
#Ari Silburt's UNet deeper
def unet_deeper(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2), )(a3)

    a4 = Convolution2D(n_filters * 8, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3P)
    a4 = Convolution2D(n_filters * 8, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a4)
    a4P = MaxPooling2D((2, 2), strides=(2, 2), )(a4)
    u = Convolution2D(n_filters * 8, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(a4P)
    u = Convolution2D(n_filters * 8, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a4, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    if k2:
        model = Model(inputs=img_input, outputs=u)
    else:
        model = Model(input=img_input, output=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',metrics=['binary_accuracy'], optimizer=optimizer)
    print(model.summary())
    return model

if __name__ == '__main__':
    #simple_resunet_upsample(256,112)#21,368,705
    unet_deeper(256,0.0001,1e-6,0.15,3,'he_normal',112)
