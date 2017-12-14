import numpy as np
import pickle
import pdb
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
from siamese_model import match_model, pose_model, identity_loss
import matplotlib.pyplot as plt
from train import get_model, load_all_imgs, posenet_generator


def plot_results(errors, save_file):
    errors.sort()
    deg_err = [np.rad2deg(x) for x in errors]
    x = []
    y = []
    for i in range(len(deg_err)):
        y.append((i+1)//len(deg_err))
        x.append(np.max(deg_err[:i+1]))

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.plot(x, y)
    plt.ylabel('Fraction of histogram')
    plt.xlabel('Angular error')
    fig.savefig(save_file)
    plt.close(fig)

    
def relative_orientation_error(q1, q2):
    q1_norm = np.linalg.norm(q1, ord=2)
    q2_norm = np.linalg.norm(q2, ord=2)
    if q1_norm > 0:
        q1 = q1/q1_norm
    if q2_norm > 0:
        q2 = q2/q2_norm
    theta_1 = 2*np.arccos(q1[0])
    theta_2 = 2*np.arccos(q2[0])

    return abs(theta_1 - theta_2)


def relative_translation(t1, t2):
	
    t1_norm = np.linalg.norm(t1, ord=2)
    t2_norm = np.linalg.norm(t2, ord=2)
    if t1_norm > 0:
        t1 = t1 / t1_norm
    if t2_norm > 0:
        t2 = t2/ t2_norm

    angle = np.arccos(np.inner(t1, t2))
    return abs(angle)


def create_test_date(img_dict, rel_quaternions, rel_translations, n_samples=1000):

    test_inp = []
    test_q = []
    test_t = []
    
    print(rel_quaternions.keys()[0])

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
            test_q.append(rel_quaternions[k2].flatten())
            test_t.append(rel_translations[k2].flatten())

    return test_inp, test_q, test_t


def test_generator(test_inp, img_dict):
    for test_pair in test_inp:
        img1 = np.reshape(img_dict[test_pair[0]], (1, 224, 224, 3))
        img2 = np.reshape(img_dict[test_pair[1]], (1, 224, 224, 3))
        yield [img1, img2]


def evaluate_posenet(model_name, dataset_path):
    ###====================== Data for testing ===========================###
    img_paths = os.listdir(dataset_path)
    test_imgs_dict = load_all_imgs(img_paths, dataset_path)
    rel_quaternions = pickle.load(open("data/rel_quaternions_test.pkl", "rb"))
    rel_translations = pickle.load(open("data/rel_translations_test.pkl", "rb"))

    test_inp, test_q, test_t = create_test_date(test_imgs_dict, rel_quaternions, rel_translations, n_samples=1000)
    model = load_model(model_name)
    pred_q, pred_t = model.predict_generator(test_generator(test_inp, test_imgs_dict), steps=999,
                                                               max_queue_size=1, workers=1,
                                                               use_multiprocessing=False, verbose=1)
    roe = []
    rte = []
    for i in range(len(pred_q)):
        roe.append(relative_orientation_error(test_q[i].flatten(), pred_q[i].flatten()))
        rte.append(relative_translation(test_t[i].flatten(), pred_t[i].flatten()))

    print('-----------------------------------------------------')
    print('ROE ....')
    print('min(ROE) = %0.3f ---- %0.3f'%(np.min(roe), np.rad2deg(np.min(roe))))
    print('max(ROE) = %0.3f ---- %0.3f'%(np.max(roe), np.rad2deg(np.max(roe))))
    print('std(ROE) = %0.3f ---- %0.3f'%(np.std(roe), np.rad2deg(np.std(roe))))    
    print('mean(ROE) = %0.3f ---- %0.3f'%(np.mean(roe), np.rad2deg(np.mean(roe))))
    print('RTE ....')
    print('min(RTE) = %0.3f ---- %0.3f'%(np.min(rte), np.rad2deg(np.min(rte))))
    print('max(RTE) = %0.3f ---- %0.3f'%(np.max(rte), np.rad2deg(np.max(rte))))
    print('std(RTE) = %0.3f ---- %0.3f'%(np.std(rte), np.rad2deg(np.std(rte))))    
    print('mean(RTE) = %0.3f ---- %0.3f'%(np.mean(rte), np.rad2deg(np.mean(rte))))    
    plot_results(roe, 'roe.png')
    plot_results(rte, 'rte.png')


# Run the script as follows:
# python evaluate.py --dataset_path='/home/sudeep/khushi/KingsCollege/seq6' --model_name='models/posenet-79.hdf5'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation utility for trained models')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='posenet', help='Type of model - posenet, matchnet, hybrid')

    args = parser.parse_args()

    evaluate_posenet(args.model_name, args.dataset_path)
