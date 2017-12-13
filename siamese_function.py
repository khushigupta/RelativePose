from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import concatenate
import tensorflow as tf


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def euclidean_distance(outputs):
    x, y, l = outputs
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    # hinge1 = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    # hinge2 = max(0, 1 - K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True)))
    #
    # return tf.cond(l == 1, hinge1, hinge2)


def eucl_dist_output_shape(shapes):
    return shapes[0][0], 1


def vgg_16():

    model = VGG16(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000)
    return model


def base_model(input_shape):

    vgg = vgg_16()

    inp = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', weights=vgg.layers[1].get_weights())(inp)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', weights=vgg.layers[2].get_weights())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', weights=vgg.layers[3].get_weights())(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', weights=vgg.layers[4].get_weights())(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', weights=vgg.layers[5].get_weights())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', weights=vgg.layers[6].get_weights())(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', weights=vgg.layers[7].get_weights())(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', weights=vgg.layers[8].get_weights())(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', weights=vgg.layers[9].get_weights())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', weights=vgg.layers[10].get_weights())(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', weights=vgg.layers[11].get_weights())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', weights=vgg.layers[12].get_weights())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', weights=vgg.layers[13].get_weights())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', weights=vgg.layers[14].get_weights())(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', weights=vgg.layers[15].get_weights())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', weights=vgg.layers[16].get_weights())(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', weights=vgg.layers[17].get_weights())(x)

    return Model(inp, x)


def match_model(input_shape, label_shape):

    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)
    labels = Input(shape=label_shape)

    base = base_model(input_shape)
    flat = Flatten()
    dense = Dense(128, activation='relu', name='fc6')

    out_left = base(input_left)
    out_left = flat(out_left)
    out_left = dense(out_left)

    out_right = base(input_right)
    out_right = flat(out_right)
    out_right = dense(out_right)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([out_left, out_right, labels])

    model = Model([input_left, input_right], distance)
    return model


def pose_model(input_shape, match_model=None):

    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)

    if match_model is None:

        base = base_model(input_shape)
        avg = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
        flat = Flatten()
        dense = Dense(4096, activation='relu', name='fc6')

        out_left = base(input_left)
        out_left = avg(out_left)
        out_left = flat(out_left)
        out_left = dense(out_left)

        out_right = base(input_right)
        out_right = avg(out_right)
        out_right = flat(out_right)
        out_right = dense(out_right)

    else:

        base = Sequential()
        for i, layer in enumerate(match_model.layers):
            if 1 <= i <= 17:
                base.add(layer)
        base.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        base.add(Flatten())
        base.add(Dense(4096, activation='relu', name='fc6'))
        out_left = base(input_left)
        out_right = base(input_right)


    output = concatenate([out_left, out_right], axis=1)
    output = Dense(4096, activation='relu', name='fc7')(output)

    R = Dense(4, activation='relu', name='R')(output)
    t = Dense(3, activation='relu', name='t')(output)

    model = Model(inputs=[input_left, input_right], outputs=[R, t])
    return model


m = match_model((64, 64, 3), (1, ))

# p = pose_model((224, 224, 3), None)
