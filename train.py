import numpy as np
import argparse
import datetime
import pickle
import os
from random import shuffle
import random
from imageio import imread
from skimage.transform import resize

import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, save_model

from siamese_function import match_model, pose_model, identity_loss


def get_model(model_type, **kwargs):

    if model_type == 'match':
        input_shape = (None, 64, 64, 3)
        label_shape = (None, 1)
        match = match_model(input_shape, label_shape)
        match.compile(optimizer='sgd',
                      loss=identity_loss,
                      metrics=['accuracy'])


    if model_type == 'pose':

        input_shape = (None, 224, 224, 3)

        if 'match_model' not in kwargs:
            pose = pose_model(input_shape)
        else:
            match_pretrained = kwargs['match_model']
            pose = pose_model(input_shape, match_pretrained)

        pose.compile(optimizer='sgd',
                     loss=['mean_squared_error', 'mean_squared_error'],
                     metrics=['accuracy'],
                     loss_weights=[1., 1])

        return pose

'''
Data generator function for yielding training images
'''
def generate(imgs, rel_quaternions, rel_translations, batch_size = 32):

    while 1:

        # Placeholders for current iteration
        x1 = np.zeros((batch_size, 224, 224, 3))
        x2 = np.zeros((batch_size, 224, 224, 3))
        y_q = np.zeros((batch_size, 4))
        y_t = np.zeros((batch_size, 3))

        for i in range(batch_size):
            img1_path = random.choice(list(imgs.keys()))
            x = list(imgs.keys())
            x.remove(img1_path)
            img2_path = random.choice(x)
            img1 = imgs[img1_path]
            img2 = imgs[img2_path]

            k1 = 'seq1/' + img1_path + ' ' + 'seq1/' + img2_path
            k2 = 'seq1/' + img2_path + ' ' + 'seq1/' + img1_path

            if k1 in rel_quaternions.keys():
                y_q[i, :] = rel_quaternions[k1].flatten()
                y_t[i, :] = rel_translations[k1].flatten()
            else:
                y_q[i, :] = rel_quaternions[k2].flatten()
                y_t[i, :] = rel_translations[k2].flatten()

        yield ([x1, x2], [y_q, y_t])

'''
Returns a (n x 224 x 224 x 3) np array.
This is because reading from img files during training is highly inefficient.
'''
def load_all_imgs(img_paths, dataset_path):
    imgs = {}
    for img_path in os.listdir(dataset_path):
        img = imread(os.path.join(dataset_path, img_path))
        img = resize(img, (224,224)) * 255 - [122.63791547, 123.32784235, 112.4143373]
        imgs[img_path] = img
    return imgs


def train(model, imgs_train, imgs_val):

    rel_quaternions = pickle.load(open("data/rel_quaternions.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations.pkl", "rb"))

    train_generator = generate(imgs_train, rel_quaternions, rel_translations, batch_size = 32)
    val_generator = generate(imgs_val, rel_quaternions, rel_translations, batch_size = 32)

    # For checkpointing
    filepath="models/posenet-{e:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    model.fit_generator(generator = train_generator,
                        steps_per_epoch = 1000,
                        validation_data = val_generator,
                        validation_steps = 10,
                        epochs=100,
                        callbacks=[checkpoint])
    return model


def evaluate():
    pass


def predict():
    pass


def main(args):

    image_paths = os.listdir(args.dataset_path)
    shuffle(image_paths)
    validation_split = 0.05
    partition = {'train':image_paths[:int(validation_split*len(image_paths))],
                 'val':image_paths[int(validation_split*len(image_paths)):]}


    imgs_train = load_all_imgs(partition['train'], args.dataset_path)
    imgs_val = load_all_imgs(partition['val'], args.dataset_path)

    model = pose_model((224, 224, 3), None)
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer,loss=['mean_squared_error', 'mean_squared_error'])

    model = train(model, imgs_train, imgs_val)


# Run the script as follows:
# python train.py --dataset_path='/home/sudeep/khushi/KingsCollege/seq1'
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Relative Pose Estimation')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset')

    args = parser.parse_args()

    main(args)
