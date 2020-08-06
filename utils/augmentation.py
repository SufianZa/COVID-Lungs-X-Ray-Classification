import random
import os
from skimage.transform import AffineTransform, warp
import numpy as np
from PIL import Image
from skimage.util import random_noise


def flip(image):
    return np.fliplr(image)


def noise_bright(image):
    res = image + 0.15
    res[res > 1] = 1
    return random_noise(res, var=0.01)


def bright(image):
    res = image + 0.15
    res[res > 1] = 1
    return res


def shear(image):
    r = random.uniform(-0.15, 0.15)
    res = warp(image, AffineTransform(shear=r), order=1, preserve_range=True, mode='wrap')
    return res


def dark(image):
    res = image - 0.15
    res[res < 0] = 0
    return res


def noise_bright_shear(image):
    r = random.uniform(-0.15, 0.15)
    res = image + 0.15
    res[res > 1] = 1
    res = warp(random_noise(res, var=0.01), AffineTransform(shear=r), order=1, preserve_range=True, mode='wrap')
    return res


def flip_shear_noise(image):
    r = random.uniform(-0.15, 0.15)
    res = warp(np.fliplr(random_noise(image, var=0.01)), AffineTransform(shear=r), order=1, preserve_range=True,
               mode='wrap')
    return res


def flip_dark(image):
    res = image - 0.15
    res[res < 0] = 0
    res = np.fliplr(res)
    return res


def noise(image):
    return random_noise(image, var=0.01)


def shear_noise_dark(image):
    r = random.uniform(-0.15, 0.15)
    res = image - 0.15
    res[res < 0] = 0
    res = warp(random_noise(res, var=0.01), AffineTransform(shear=r), order=1, preserve_range=True, mode='wrap')
    return res


def flip_shear(image):
    r = random.uniform(-0.15, 0.15)
    res = warp(np.fliplr(image), AffineTransform(shear=r), order=1, preserve_range=True, mode='wrap')
    return res


def flip_noise(image):
    return np.fliplr(random_noise(image, var=0.01))


STRATEGY_PICKER = {
    'flip': flip,
    'noise_bright': noise_bright,
    'bright': bright,
    'shear': shear,
    'dark': dark,
    'noise_bright_shear': noise_bright_shear,
    'flip_shear_noise': flip_shear_noise,
    'flip_dark': flip_dark,
    'noise': noise,
    'shear_noise_dark': shear_noise_dark,
    'flip_shear': flip_shear,
    'flip_noise': flip_noise
}


def save_augmentation(image, image_name, label, features, data, dst_dir, aug_num=12):
    strategies = list(STRATEGY_PICKER.keys())
    random.shuffle(strategies)
    strategy_arr = strategies[:aug_num]
    for strategy in strategy_arr:
        res = STRATEGY_PICKER[strategy](image)
        result = Image.fromarray((res * 255).astype(np.uint8))
        result.save(os.path.join(dst_dir, 'Aug' + strategy + '_' + image_name))
        data['paths'].append('Aug' + strategy + '_' + image_name)
        data['features'].append(features)
        data['labels'].append(label)
