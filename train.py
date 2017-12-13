import argparse
import numpy as np
import pickle
import os
from random import shuffle
import random
from imageio import imread
from skimage.transform import resize

import keras
from keras.callbacks import ModelCheckpoint

from siamese_model import match_model, pose_model, hybrid_model, identity_loss
from match_net import matchnet_gen

def get_model(model_type, **kwargs):

    optimizer = keras.optimizers.Adam(lr=0.001)

    if model_type == 'match':

        match_shape = (64, 64, 3)
        label_shape = (1, )
        model = match_model(match_shape, label_shape)
        model.compile(optimizer='sgd',
                      loss=identity_loss,
                      metrics=['accuracy'])

        model.compile(optimizer, loss=identity_loss)

    if model_type == 'pose':

        pose_shape = (224, 224, 3)

        if 'match_model' not in kwargs:
            model = pose_model(pose_shape)
        else:
            match_pretrained = kwargs['match_model']
            model = pose_model(pose_shape, match_pretrained)
            model.compile(optimizer, loss=['mean_squared_error', 'mean_squared_error'])

    if model_type == 'hybrid':

        pose_shape = (224, 224, 3)
        match_shape = (64, 64, 3)
        label_shape = (1,)

        model = hybrid_model(pose_shape, match_shape, label_shape)
        model.compile(optimizer, loss=['mean_squared_error', 'mean_squared_error', identity_loss])

    return model


def posenet_generator(imgs, rel_quaternions, rel_translations, batch_size=32):
    ''' Data generator function for yielding training images '''

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
            x1[i, ...] = imgs[img1_path]
            x2[i, ...] = imgs[img2_path]

            k1 = 'seq1/' + img1_path + ' ' + 'seq1/' + img2_path
            k2 = 'seq1/' + img2_path + ' ' + 'seq1/' + img1_path

            if k1 in rel_quaternions.keys():
                y_q[i, :] = rel_quaternions[k1].flatten()
                y_t[i, :] = rel_translations[k1].flatten()
            else:
                y_q[i, :] = rel_quaternions[k2].flatten()
                y_t[i, :] = rel_translations[k2].flatten()

        yield ([x1, x2], [y_q, y_t])


def load_all_imgs(img_paths, dataset_path):
    '''
    Returns a (n x 224 x 224 x 3) np array.
    This is because reading from img files during training is highly inefficient.
    '''
    imgs = {}
    for img_path in img_paths:
        img = imread(os.path.join(dataset_path, img_path))
        img = resize(img, (224, 224)) * 255 - [122.63791547, 123.32784235, 112.4143373]
        imgs[img_path] = img
    return imgs


def train_posenet(dataset_path, validation_split=0.05):
    ''' Training implementation for PoseNet as of now'''

    ###====================== Fetch all images ===========================###
    image_paths = os.listdir(dataset_path)
    shuffle(image_paths)
    partition = {'train': image_paths[:int(validation_split*len(image_paths))],
                 'val': image_paths[int(validation_split*len(image_paths)):]}
    imgs_train = load_all_imgs(partition['train'], dataset_path)
    imgs_val = load_all_imgs(partition['val'], dataset_path)

    ###====================== GT for training ===========================###
    rel_quaternions = pickle.load(open("data/rel_quaternions.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations.pkl", "rb"))

    ###====================== Generators ===========================###
    train_generator = posenet_generator(imgs_train, rel_quaternions,
                                        rel_translations, batch_size=32)
    val_generator = posenet_generator(imgs_val, rel_quaternions,
                                      rel_translations, batch_size=32)


    ###====================== Train model ===========================###
    filepath = "models/posenet-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')
    model = get_model('pose', match_model=None)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=100,
                        validation_data=val_generator,
                        validation_steps=5,
                        epochs=100,
                        callbacks=[checkpoint])
    return model

def train_matchnet(dataset_path, validation_split=0.05):

    ###====================== Load train + val matches ===========================###
    matches = pickle.load(open("data/matches_clean_2.pkl", "rb"))
    shuffle(matches)
    partition = {'train': matches[:int(validation_split*len(matches))],
                 'val': matches[int(validation_split*len(matches)):]}

    ###====================== Generators ===========================###
    train_generator = matchnet_gen(partition['train'],
                                   batch_size=10, patch_size=64, pos_ratio=0.3)
    train_generator = matchnet_gen(partition['val'],
                                   batch_size=10, patch_size=64, pos_ratio=0.3)

    filepath = "models/matchnet{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')
    model = get_model('match', match_model=None)


def evaluate():
    pass


def predict():
    pass


def main(args):

    if args.model == 'posenet':
        train_posenet(args.dataset_path)
    elif args.model == 'matchnet':
        train_matchnet(args.dataset_path)
    elif args.model == 'hybrid':
        print('Model not implemented!')
    else:
        print('Invalid model. Options are posenet, matchnet, hybrid')



# Run the script as follows:
# python train.py --dataset_path='/home/sudeep/khushi/KingsCollege/seq1' --model=posenet
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Relative Pose Estimation')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset')
    parse.add_argument('--model', type=str, default='posenet', help='Type of model - posenet, matchnet, hybrid')

    args = parser.parse_args()

    main(args)
