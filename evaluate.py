import numpy as np
import pickle
import os
from random import shuffle
import random
from imageio import imread
from skimage.transform import resize
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import datetime
from keras import optimizers
from keras.models import load_model, save_model
from siamese_function import match_model, pose_model, identity_loss

from train import get_model, generate, load_all_imgsm posenet_generator


def relative_orientation_error(q1, q2):

    q1 = q1 / np.linalg.norm(q1, ord=2)
    q2 = q2 / np.linalg.norm(q2, ord=2)
    theta_1 = 2*np.arccos(q1[0])
    theta_2 = 2*np.arccos(q2[0])

    return abs(theta_1 - theta_2)


def relative_translation(t1, t2):

    t1 = t1 / np.linalg.norm(t1, ord=2)
    t2 = t2 / np.linalg.norm(t2, ord=2)
    angle = np.arccos(np.inner(t1, t2))

    return abs(angle)


def create_test_date(img_dict, rel_quaternions, rel_translations, n_samples=1000)

    test_inp = []
    test_q = []
    test_t = []

    for i in range(n_samples):
        img1_path = random.choice(list(img_dict.keys()))
        x = list(img_dict.keys())
        x.remove(img1_path)
        img2_path = random.choice(x)
        test_inp.append([img1_path, img2_path])

        k1 = 'seq6/' + img1_path + ' ' + 'seq6/' + img2_path
        k2 = 'seq6/' + img2_path + ' ' + 'seq6/' + img1_path

        if k1 in rel_quaternions.keys():
            test_q.append(rel_quaternions[k1].flatten())
            test_t.append(rel_translations[k1].flatten())
        else:
            test_q.append(rel_quaternions[k1].flatten())
            test_t.append(rel_translations[k1].flatten())

    return test_inp, test_q, test_t


def test_generator(test_inp, img_dict):
    for test_pair in range(test_inp):
        img1 = np.reshape(img_dict[test_pair[0]], (1, 224, 224, 3))
        img2 = np.reshape(img_dict[test_pair[1]], (1, 224, 224, 3))
        yield [img1, img2]


def evaluate_posenet(model_name, dataset_path):
    ###====================== Data for testing ===========================###
    img_paths = os.listdir(dataset_path)
    test_imgs_dict = load_all_imgs(img_paths, dataset_path)
    rel_quaternions = pickle.load(open("data/rel_quaternions.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations.pkl", "rb"))

    test_inp, test_q, test_t = create_test_date(test_imgs_dict, rel_quaternions, rel_translations, n_samples=1000)

    pred_q, pred_t = model.predict_generator(test_generator(test_inp, test_imgs_dict), steps=1000,
                                                               max_queue_size=10, workers=1,
                                                               use_multiprocessing=False)

# Run the script as follows:
# python evaluate.py --dataset_path='/home/sudeep/khushi/KingsCollege/seq6' --model_name='models/posenet-79.hdf5'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation utility for trained models')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='posenet', help='Type of model - posenet, matchnet, hybrid')

    args = parser.parse_args()

    evaluate(args.model_name, args.dataset_path)
