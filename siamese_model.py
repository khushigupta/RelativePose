from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import concatenate
import tensorflow as tf


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def hinge_single(single_input):

    x, y, l = single_input

    def hinge1(a, b):
        return K.sqrt(K.sum(K.square(a - b), axis=0))

    def hinge2(a, b):
        return tf.maximum(0.0, 1 - hinge1(a, b))

    output = tf.cond(tf.equal(l[0], 1), lambda: hinge1(x, y), lambda: hinge2(x, y))
    return output


def hinge_batch(inputs):
    return tf.map_fn(fn=hinge_single, elems=(inputs[0], inputs[1], inputs[2]), dtype=tf.float32)


def hinge_output_shape(shapes):
    return shapes[0][0], 1


def vgg_16():

    model = VGG16(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000)
    return model


def base_model():

    vgg = vgg_16()
    inp = Input(shape=(64, 64, 3))
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

    base = base_model()
    flat = Flatten()
    dense1 = Dense(512, activation='relu', name='fc6')
    dense2 = Dense(128, activation='relu', name='fc7')

    out_left = base(input_left)
    out_left = flat(out_left)
    out_left = dense1(out_left)
    out_left = dense2(out_left)

    out_right = base(input_right)
    out_right = flat(out_right)
    out_right = dense1(out_right)
    out_right = dense2(out_right)

    distance = Lambda(hinge_batch, output_shape=hinge_output_shape)([out_left, out_right, labels])

    model = Model(inputs=[input_left, input_right, labels], outputs=distance)
    return model


def pose_model(input_shape, match_model=None):

    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)

    if match_model is None:
        base = base_model()
    else:
        base = match_model.layers[2]  # the base model

    # avg = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    avg = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    flat = Flatten()
    dense = Dense(2048, activation='relu', name='fc6')

    out_left = base(input_left)
    out_left = avg(out_left)
    out_left = flat(out_left)
    out_left = dense(out_left)

    out_right = base(input_right)
    out_right = avg(out_right)
    out_right = flat(out_right)
    out_right = dense(out_right)

    output = concatenate([out_left, out_right], axis=1)
    output = Dense(512, activation='relu', name='fc7')(output)

    R = Dense(4, activation='tanh', name='R')(output)
    t = Dense(3, activation='tanh', name='t')(output)

    model = Model(inputs=[input_left, input_right], outputs=[R, t])
    return model


def hybrid_model(pose_shape, match_shape, label_shape):

    pose_left = Input(shape=pose_shape)
    pose_right = Input(shape=pose_shape)
    match_left = Input(shape=match_shape)
    match_right = Input(shape=match_shape)
    match_labels = Input(shape=label_shape)

    base = base_model()

    # Pose branch
    max_pose = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    flat_pose = Flatten()
    dense_pose = Dense(2048, activation='relu', name='pose_fc6')

    pose_out_left = base(pose_left)
    pose_out_left = max_pose(pose_out_left)
    pose_out_left = flat_pose(pose_out_left)
    pose_out_left = dense_pose(pose_out_left)

    pose_out_right = base(pose_right)
    pose_out_right = max_pose(pose_out_right)
    pose_out_right = flat_pose(pose_out_right)
    pose_out_right = dense_pose(pose_out_right)

    output = concatenate([pose_out_left, pose_out_right], axis=1)
    output = Dense(512, activation='relu', name='pose_fc7')(output)

    R = Dense(4, activation='tanh', name='R')(output)
    t = Dense(3, activation='tanh', name='t')(output)

    # Match branch
    flat_match = Flatten()
    dense_match1 = Dense(512, activation='relu', name='match_fc6')
    dense_match2 = Dense(128, activation=None, name='match_fc7')

    match_out_left = base(match_left)
    match_out_left = flat_match(match_out_left)
    match_out_left = dense_match1(match_out_left)
    match_out_left = dense_match2(match_out_left)

    match_out_right = base(match_right)
    match_out_right = flat_match(match_out_right)
    match_out_right = dense_match1(match_out_right)
    match_out_right = dense_match2(match_out_right)

    distance = Lambda(hinge_batch, output_shape=hinge_output_shape)([match_out_left, match_out_right, match_labels])

    model = Model(inputs=[pose_left, pose_right, match_left, match_right, match_labels], outputs=[R, t, distance])
    return model

