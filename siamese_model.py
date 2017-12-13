from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import keras as K
from keras.models import Sequential, Model




def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def euclidean_distance(outputs):
    x, y, l = outputs

    if l == 1:
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    else:
        return max(0, 1 - K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True)))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


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
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i, layer in enumerate(vgg.layers):
        if 1 <= i <= 17:
            model.add(layer)

    return model


def match_model(input_shape, label_shape):

    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)
    labels = Input(shape=label_shape)

    base = base_model(input_shape)
    base.add(Flatten())
    base.add(Dense(128, activation='relu', name='fc6'))


    out_left = base(input_left)
    out_right = base(input_right)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([out_left, out_right, labels])

    model = Model([input_left, input_right], distance)
    return model


def pose_model(input_shape, match_model=None):

    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)

    if match_model is None:
        base = base_model(input_shape)

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

    output = K.layers.concatenate([out_left, out_right], axis=1)
    output = Dense(4096, activation='relu', name='fc7')(output)

    R = Dense(4, activation='relu', name='R')(output)
    t = Dense(3, activation='relu', name='t')(output)

    model = Model([input_left, input_right], output=[R, t])
    return model
