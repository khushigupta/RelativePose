import random
import os
import pdb
import numpy as np

from imageio import imread

class Match(object):
    def __init__(self, img1_path, img2_path, locs):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.locs = locs

def get_crop(img, center, patch_size=64):
    '''
    Returns a patch of size 64x64 by default centered at center:
    - center : tuple of (x,y)
    If center is near the corners, None is returned
    '''

    height, width = img.shape[0], img.shape[1]
    y_pos, x_pos = center
    if x_pos + patch_size/2 > height or x_pos - patch_size/2 < 0:
        return None
    if y_pos + patch_size/2 > width or y_pos - patch_size/2 < 0:
        return None
    return img[int(x_pos-patch_size/2):int(x_pos+patch_size/2),
               int(y_pos-patch_size/2):int(y_pos+patch_size/2), :]

def get_negative_crop(img, center_to_avoid, patch_size=64):
    '''
    Returns a patch of size 64x64 by default which does not include a given
    patch centered at center:
    - center_to_avoid : tuple of (x,y)
    '''
    height, width = img.shape[0], img.shape[1]
    y_pos, x_pos = center_to_avoid
    y_rand = int(np.random.uniform(patch_size/2, width-patch_size/2))
    x_rand = int(np.random.uniform(patch_size/2, height-patch_size/2))

    if x_pos - patch_size/2 <= x_rand <= x_pos + patch_size/2:
        return get_negative_crop(img, center_to_avoid, patch_size)
    if y_pos - patch_size/2 <= y_rand <= y_pos + patch_size/2:
        return get_negative_crop(img, center_to_avoid, patch_size)

    return img[int(x_rand-patch_size/2):int(x_rand+patch_size/2),
               int(y_rand-patch_size/2):int(y_rand+patch_size/2), :]

def matchnet_gen(matches, dataset_path, batch_size=10, patch_size=64, pos_ratio=0.3,
                 mode='yield'):
    ''' Generator for the matchnet architecture '''

    while 1:
        # Placeholders for current iteration
        patches1 = np.zeros((batch_size, patch_size, patch_size, 3))
        patches2 = np.zeros((batch_size, patch_size, patch_size, 3))
        labels = np.zeros(10)

        # One image pair per iteration
        match = random.choice(matches)
        img1 = imread(os.path.join(dataset_path, os.path.basename(match.img1_path))) - [122.63791547, 123.32784235, 112.4143373]
        img2 = imread(os.path.join(dataset_path, os.path.basename(match.img2_path))) - [122.63791547, 123.32784235, 112.4143373]

        for i in range(batch_size):
            loc = random.choice(match.locs)
            patches1[i, ...] = get_crop(img1, loc[0], patch_size=64)

            # Choose negative or positive sample
            prob = np.random.uniform(0.0, 1.0)

            # Positive case
            if prob <= pos_ratio:
                patches2[i, ...] = get_crop(img2, loc[1], patch_size=64)
                labels[i] = 1

            # Negative case
            else:
                patches2[i, ...] = get_negative_crop(img2, center_to_avoid=loc[1], patch_size=64)
                labels[i] = 0
        if mode == 'return':
            return [patches1, patches2, labels], np.zeros((batch_size))
        else:
            yield [patches1, patches2, labels], np.zeros((batch_size))
