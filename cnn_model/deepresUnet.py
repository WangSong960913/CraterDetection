from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers.core import  Reshape
from keras.layers import Input, Conv2DTranspose,Dropout, BatchNormalization, Conv2D, MaxPooling2D,\
    concatenate, Activation,add,UpSampling2D
# batchnormalization 后激活
def BatchActivate(x):
    x = BatchNormalization(axis=3, momentum=0.01)(x)
    x = Activation('relu')(x)
    return x
# 卷积block
def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    if activation == True:
        x = BatchActivate(x)
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    return x
#参差卷积块
def residual_block(blockInput, num_filters=16,stride=1, filters_change=True):
    downsample=blockInput
    if stride==2:
        downsample=Conv2D(num_filters,(1,1),strides=(2, 2))(blockInput)
        x2 = convolution_block(blockInput, num_filters, (3, 3),strides=(2, 2))
    else:
        x2 = convolution_block(blockInput, num_filters, (3, 3),activation=False)
    x3 = convolution_block(x2, num_filters, (3, 3))
    if filters_change and stride==1:
        downsample=Conv2D(num_filters,(1,1),strides=(1, 1))(blockInput)
        x = add([downsample, x3])
    else:
        x = add([downsample, x3])
    return x
#simple res-Unet with two residual_block
def deep_residual_Unet(dim,start_filter,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #level1
    conv1 = residual_block(input_layer, start_filter * 1,stride=1)
    #level2
    conv2 = residual_block(conv1, start_filter * 2,stride=2)
    # level3
    conv3 = residual_block(conv2, start_filter * 4, stride=2)
    #level4
    conv4 = residual_block(conv3, start_filter * 8, stride=2)
    #level5
    conv4_half = Conv2D(start_filter * 4, (1, 1), strides=1)(conv4)
    up_5=UpSampling2D((2,2))(conv4_half)
    concate5 = concatenate([conv3, up_5])
    concate5=BatchActivate(concate5)
    conv5 = residual_block(concate5, start_filter * 4)
    #level6
    up_6=UpSampling2D((2,2))(conv5)
    concate6 = concatenate([conv2, up_6])
    concate6=BatchActivate(concate6)
    conv6 = residual_block(concate6, start_filter * 2)
    #level7
    up_7=UpSampling2D((2,2))(conv6)
    concate7 = concatenate([conv1, up_7])
    concate7=BatchActivate(concate7)
    conv7 = residual_block(concate7, start_filter)
    #out
    conv8=Conv2D(1, (1,1), strides=1)(conv7)
    output_layer = Activation('sigmoid')(conv8)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
    return model
#simple res-Unet
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
def resnet_34(dim,init_filter ):
    inpt = Input(shape=(dim, dim, 1))
    # encoder
    # conv1
    x_1 = identity_Block(inpt, nb_filter=init_filter, kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    # conv2_x
    x_2 = identity_Block(x_1, nb_filter=init_filter*2, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    # conv3_x
    x_3 = identity_Block(x_2, nb_filter=init_filter*4, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    # bridgeconv_4
    x_4 = identity_Block(x_3, nb_filter=init_filter*8, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    # decoder
    # conv_5

    up_5=UpSampling2D((2,2))(x_4)
    x_6 = concatenate([x_3, up_5])
    x_7 = identity_Block(x_6, nb_filter=init_filter*4, kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    # conv_6
    x_8 = UpSampling2D((2,2))(x_7)
    x_9 = concatenate([x_8, x_2])
    x_10 = identity_Block(x_9, nb_filter=init_filter*2, kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    # conv_7
    x_11 = UpSampling2D((2,2))(x_10)
    x_12=concatenate([x_11, x_1])
    x_13 = identity_Block(x_12, nb_filter=init_filter, kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    conv8=Conv2D(1, (1,1), strides=1)(x_13)
    output_layer = Activation('sigmoid')(conv8)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(inpt, output_layer)
    c = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
    return model

if __name__ == '__main__':
    #simple_resunet_upsample(256,112)#21,368,705
    #deep_residual_Unet(256,112)
    resnet_34(256,112)
