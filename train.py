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
def generate(img_paths, rel_quaternions, rel_translations, batch_size = 32,
             dataset_path='/Users/animesh/Downloads/geometry_images/KingsCollege/seq1'):

    while 1:

        # Placeholders for current iteration
        x1 = np.zeros((batch_size, 224, 224, 3))
        x2 = np.zeros((batch_size, 224, 224, 3))
        y_q = np.zeros((batch_size, 4))
        y_t = np.zeros((batch_size, 3))

        for i in range(batch_size):
            img1_path = random.choice(img_paths)
            x = list(img_paths)
            x.remove(img1_path)
            img2_path = random.choice(x)
            img1 = imread(os.path.join(dataset_path, img1_path))
            img2 = imread(os.path.join(dataset_path, img2_path))
            x1[i, ...] = resize(img1, (224,224))
            x2[i, ...] = resize(img2, (224, 224))

            k1 = 'seq1/' + img1_path + ' ' + 'seq1/' + img2_path
            k2 = 'seq1/' + img2_path + ' ' + 'seq1/' + img1_path

            if k1 in rel_quaternions.keys():
                y_q[i, :] = rel_quaternions[k1].flatten()
                y_t[i, :] = rel_translations[k1].flatten()
            else:
                y_q[i, :] = rel_quaternions[k2].flatten()
                y_t[i, :] = rel_translations[k2].flatten()

        yield ([x1, x2], [y_q, y_t])

def train(model, image_paths):

    rel_quaternions = pickle.load(open("data/rel_quaternions.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations.pkl", "rb"))
    filepath="posenet-{e:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Split matches into train and validation
    shuffle(image_paths)
    validation_split = 0.05
    partition = {'train':image_paths[:int(validation_split*len(image_paths))],
              'val':image_paths[int(validation_split*len(image_paths)):]}

    train_generator = generate(partition['train'], rel_quaternions, rel_translations, batch_size = 32,
                      dataset_path='/Users/animesh/Downloads/geometry_images/KingsCollege/seq1')
    val_generator = generate(partition['val'], rel_quaternions, rel_translations, batch_size = 32,
                    dataset_path = '/Users/animesh/Downloads/geometry_images/KingsCollege/seq1')

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


def get_name():

    time_str = datetime.now().strftime("%M_%H_%d")
    model_name = '{}/{}_{}.h5'.format(args.path, args.model_type, time_str)
    return model_name


def main():

    rel_quaternions = pickle.load(open("data/rel_quaternions.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations.pkl", "rb"))
    dataset_path = '/Users/animesh/Downloads/geometry_images/KingsCollege/seq1'
    image_paths = os.listdir(dataset_path)

    model = pose_model((224, 224, 3), None)
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer,loss=['mean_squared_error', 'mean_squared_error'])

    model = train(model, image_paths)



    # if args.model_type == 'match':
    #     model = get_model('match')
    #
    # else:
    #     if args.pretrain_path is not None:
    #         match_trained = load_model(args.pretrain_path)
    #         model = get_model('match', pretrain=match_trained)
    #     else:
    #         model = get_model('match')
    #
    # if args.train:
    #     trained_model = train(model)
    #     save_model(get_name(), trained_model)
    #
    #     print('Finished training!')
    #
    # if args.evaluate:
    #     model = load_model(args.e_model)
    #     evaluate(model)
    #     print('Finished Evaluating!')
    #
    # if args.predict:
    #     model = load_model(args.p_model)
    #     predict(model)
    #     print('Finished Predicting!')


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='PyTorch Training')
    #
    # parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')
    # parser.add_argument('--train', type=int, default=True, help='Train model')
    # parser.add_argument('--model_type', type=str, default='', help='Train Pose/SIFT model')
    # parser.add_argument('--pretrain_path', type=str, default='', help='Path to pretrained SIFT model')
    #
    # parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    # parser.add_argument('--batch_size', type=int, default=32, help='Batch Size (default: 32)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    #
    # parser.add_argument('--path', type=str, default='', help='Path to store trained models')
    # parser.add_argument('--log_path', type=str, default='', help='Path to store logs')
    #
    # parser.add_argument('--evaluate', type=int, default=False, help='Evaluate model')
    # parser.add_argument('--e_model', type=int, default=False, help='Model to evaluate')
    #
    # parser.add_argument('--predict', type=int, default=False, help='Get predictions')
    # parser.add_argument('--p_model', type=int, default=False, help='Model to use for predictions')
    #
    # args = parser.parse_args()

    main()
