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

from train import get_model, generate


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


def evaluate(model):

    test_generator = generate(partition['test'], rel_quaternions, rel_translations, batch_size=32,
                             dataset_path='/Users/animesh/Downloads/geometry_images/KingsCollege/seq1')


    [rel_q, rel_t] = model.predict(x, batch_size=None, verbose=0, steps=None)





