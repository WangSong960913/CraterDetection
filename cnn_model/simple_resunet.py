from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers.core import  Reshape
from keras.layers import Input, Conv2DTranspose,Dropout, BatchNormalization, Conv2D, MaxPooling2D,\
    concatenate, Activation,add,UpSampling2D
# batchnormalization +relu
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
    x1 = BatchActivate(blockInput)
    x2 = convolution_block(x1, num_filters, (3, 3))
    x3 = convolution_block(x2, num_filters, (3, 3), activation=False)
    if filters_change:
        x = add([x2, x3])
    else:
        x = add([x1, x3])
    if batch_activate:
        x = BatchActivate(x)
    return x
#simple res-Unet with two residual_block
def simple_resunet_two_resblock(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, start_filter * 4)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1)
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
#simple res-Unet
def simple_resunet(dim,start_filter,DropoutRatio=0.3,lr=0.0001,class_num=1):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(class_num, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1
def simple_resunet_sallite(dim,start_filter,DropoutRatio=0.3,lr=0.0001,class_num=1):
    input_layer = Input(batch_shape=(None, dim, dim, 3))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(class_num, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    #output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1
def simple_resunet_only_resblock(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = residual_block(input_layer, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    #down_unit_2
    conv2 = residual_block(pool1, start_filter * 2, True,True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    #down_unit_3
    conv3 = residual_block(pool2, start_filter * 4, True,True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, start_filter * 4, True)

    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True,True)

    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True,True)

    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True,True)

    #output
    output_layer = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(uconv1)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
def simple_resunet_only_tworesblock(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = residual_block(input_layer, start_filter * 1, True,True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    #down_unit_2
    conv2 = residual_block(pool1, start_filter * 2, True,True)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    #down_unit_3
    conv3 = residual_block(pool2, start_filter * 4, True,True)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    #convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(pool3, start_filter * 4, True)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True,True)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True,True)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True,True)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(uconv1)
    output_layer = Reshape((dim, dim))(output_layer)
    model = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model.summary()
    return model
def simple_resunet_deeper(dim,start_filter,DropoutRatio=0.3,lr=0.0001):

    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    #down_unit_4
    conv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_filter * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_filter * 8, True)

    #up_unit_1
    deconv4 = Conv2DTranspose(start_filter * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    uconv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_filter * 8, True)

    #up_unit_2
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)

    #up_unit_3
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)

    #up_unit_4
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1
def simple_resunet_deeper_2(dim,start_filter,DropoutRatio=0.3,lr=0.0001):

    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    #down_unit_4
    conv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_filter * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    #down_unit_4
    conv5 = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(pool4)
    conv5 = residual_block(conv5,start_filter * 16, True)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(DropoutRatio)(pool5)
    # Middle
    convm = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(pool5)
    convm = residual_block(convm, start_filter * 16, True)

    deconv5 = Conv2DTranspose(start_filter * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(DropoutRatio)(uconv5)
    uconv5 = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(uconv5)
    uconv5 = residual_block(uconv5,start_filter * 16, True)
    #up_unit_1
    deconv4 = Conv2DTranspose(start_filter * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    uconv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_filter * 8, True)

    #up_unit_2
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)

    #up_unit_3
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)

    #up_unit_4
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1
def simple_resunet_deeper_8(dim,start_filter,DropoutRatio=0.3,lr=0.0001):

    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    #down_unit_4
    conv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_filter * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)


        #down_unit_5
    conv5 = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(pool4)
    conv5 = residual_block(conv5,start_filter * 16, True)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(DropoutRatio)(pool5)
        #down_unit_6
    conv6 = Conv2D(start_filter * 32, (3, 3), activation=None, padding="same")(pool5)
    conv6 = residual_block(conv6,start_filter * 32, True)
    pool6 = MaxPooling2D((2, 2))(conv6)
    pool6 = Dropout(DropoutRatio)(pool6)
        #down_unit_7
    conv7 = Conv2D(start_filter * 64, (3, 3), activation=None, padding="same")(pool6)
    conv7 = residual_block(conv7,start_filter * 64, True)
    pool7 = MaxPooling2D((2, 2))(conv7)
    pool7 = Dropout(DropoutRatio)(pool7)
        #down_unit_8
    conv8 = Conv2D(start_filter * 128, (3, 3), activation=None, padding="same")(pool7)
    conv8 = residual_block(conv8,start_filter * 128, True)
    pool8 = MaxPooling2D((2, 2))(conv8)
    pool8 = Dropout(DropoutRatio)(pool8)
        #down_unit_9
    # conv9 = Conv2D(start_filter * 256, (3, 3), activation=None, padding="same")(pool8)
    # conv9 = residual_block(conv9,start_filter * 256, True)
    # pool9 = MaxPooling2D((2, 2))(conv9)
    # pool9 = Dropout(DropoutRatio)(pool9)
    # Middle
    convm = Conv2D(start_filter * 256, (3, 3), activation=None, padding="same")(pool8)
    convm = residual_block(convm, start_filter * 256, True)

    # deconv9 = Conv2DTranspose(start_filter * 256, (3, 3), strides=(2, 2), padding="same")(convm)
    # uconv9 = concatenate([deconv9, conv9])
    # uconv9 = Dropout(DropoutRatio)(uconv9)
    # uconv9 = Conv2D(start_filter * 256, (3, 3), activation=None, padding="same")(uconv9)
    # uconv9 = residual_block(uconv9,start_filter * 256, True)

    deconv8 = Conv2DTranspose(start_filter * 128, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv8 = concatenate([deconv8, conv8])
    uconv8 = Dropout(DropoutRatio)(uconv8)
    uconv8 = Conv2D(start_filter * 128, (3, 3), activation=None, padding="same")(uconv8)
    uconv8 = residual_block(uconv8,start_filter * 128, True)

    deconv7 = Conv2DTranspose(start_filter * 64, (3, 3), strides=(2, 2), padding="same")(uconv8)
    uconv7 = concatenate([deconv7, conv7])
    uconv7 = Dropout(DropoutRatio)(uconv7)
    uconv7 = Conv2D(start_filter * 64, (3, 3), activation=None, padding="same")(uconv7)
    uconv7 = residual_block(uconv7,start_filter * 64, True)

    deconv6 = Conv2DTranspose(start_filter * 32, (3, 3), strides=(2, 2), padding="same")(uconv7)
    uconv6 = concatenate([deconv6, conv6])
    uconv6 = Dropout(DropoutRatio)(uconv6)
    uconv6 = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(uconv6)
    uconv6 = residual_block(uconv6,start_filter * 16, True)

    deconv5 = Conv2DTranspose(start_filter * 16, (3, 3), strides=(2, 2), padding="same")(uconv6)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(DropoutRatio)(uconv5)
    uconv5 = Conv2D(start_filter * 16, (3, 3), activation=None, padding="same")(uconv5)
    uconv5 = residual_block(uconv5,start_filter * 16, True)
    #up_unit_1
    deconv4 = Conv2DTranspose(start_filter * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    uconv4 = Conv2D(start_filter * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_filter * 8, True)

    #up_unit_2
    deconv3 = Conv2DTranspose(start_filter * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)

    #up_unit_3
    deconv2 = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)

    #up_unit_4
    deconv1 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1
def simple_resunet_upsample(dim,start_filter,DropoutRatio=0.3,lr=0.0001):
    input_layer = Input(batch_shape=(None, dim, dim, 1))
    #down_unit_1
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_filter * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)
    #down_unit_2
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_filter * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    #down_unit_3
    conv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_filter * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    # Middle
    convm = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, start_filter * 4, True)
    #up_unit_1
    deconv3 = UpSampling2D()(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv2D(start_filter * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_filter * 4, True)
    #up_unit_2
    deconv2 = UpSampling2D()(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_filter * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_filter * 2, True)
    #up_unit_3
    deconv1 = UpSampling2D()(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_filter * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_filter * 1, True)
    #output
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    output_layer = Reshape((dim, dim))(output_layer)
    model1 = Model(input_layer, output_layer)
    c = Adam(lr=lr)
    model1.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=c)
    model1.summary()
    return model1

if __name__ == '__main__':
    #simple_resunet_upsample(256,112)#21,368,705
    simple_resunet(256,112)
