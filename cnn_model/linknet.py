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
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x
#参差卷积块
def residual_block(blockInput, num_filters=16, batch_activate=False,filters_change=False):
    #x1 = BatchActivate(blockInput)
    x2 = convolution_block(blockInput, num_filters, (3, 3))
    x3 = convolution_block(x2, num_filters, (3, 3))
    if filters_change:
        shortcut = convolution_block(blockInput, num_filters,(3, 3))
        shortcut=BatchActivate(shortcut)
        x = add([shortcut, x3])
    else:
        x = add([blockInput, x3])
    # if batch_activate:
    #     x = BatchActivate(x)
    return x
def encoder_block(input,filter):
    conv1=convolution_block(input,filter,(3,3))
    conv2=convolution_block(conv1,filter,(3,3))
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
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

def linknet(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1,filters_change=False)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    #conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(pool1, start_filter * 2,filters_change=True)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    #conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(pool2, start_filter * 4,filters_change=True)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    #convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(pool3, start_filter * 4,filters_change=True)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    #uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4,filters_change=True)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    #uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2,filters_change=True)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    #uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1,filters_change=True)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
    return model
def linknet_new(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1,filters_change=False)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    #conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(pool1, start_filter * 2,filters_change=True)
    conv2 = residual_block(conv2, start_filter * 2, True)
    # pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    #conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(pool2, start_filter * 4,filters_change=True)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    #convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(pool3, start_filter * 4,filters_change=True)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    #uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4,filters_change=True)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    #uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2,filters_change=True)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    #uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1,filters_change=True)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
    return model
# def decoder(inpt,in_channels,n_filters):
#     x=Conv2d_BN(inpt,in_channels // 4,(1,1))
#     x=

def resnet_34(width,height,channel,classes):
    inpt = Input(batch_shape=(width, height, channel))
    #inpt = Input(batch_shape=(None, dim, dim, 1))
    #x = ZeroPadding2D((3, 3))(inpt)
    #conv1
    x = Conv2d_BN(inpt, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    #conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    e1=x
    #conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    e2=x
    #conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    e3=x
    #conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    e4=x

    model = Model(inputs=inpt, outputs=x)
    return model

if __name__ == '__main__':
    #simple_resunet_upsample(256,112)#21,368,705
    linknet(256,112)
