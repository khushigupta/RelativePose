from siamese_model import vgg_16
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import keras as K


class Model:

    def __init__(self):
        self.vgg = vgg_16()


    def base_model(self, input_shape):

        out = Input(shape=input_shape)

        for i, layer in enumerate(self.vgg.layers):
            if 1 <= i <= 17:
                out = layer(out)
        return out


    def pose_model(self, input):

        output = self.base_model(input)
        output = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(output)
        output = Flatten()(output)
        output = Dense(4096, activation='relu', name='fc6')(output)
        return output


    def pose_model_forward(self, input_left, input_right):

        out_left = self.pose_model(input_left)
        out_right = self.pose_model(input_right)

        output = K.layers.concatenate([out_left, out_right], axis=1)
        output = Dense(4096, activation='relu', name='fc7')(output)

        R = Dense(4, activation='relu', name='R')(output)
        t = Dense(3, activation='relu', name='t')(output)

        return [R, t]


    def match_model(self, input):

        output = self.base_model(input)
        output = Flatten()(output)
        output = Dense(128, activation='relu', name='fc6')(output)
        return output


    def match_model_forward(self, input_left, input_right):

        out_left = self.match_model(input_left)
        out_right = self.match_model(input_right)
        return [out_left, out_right]













