import random
import os
from skimage.transform import AffineTransform, warp, rotate
import numpy as np
from PIL import Image
from skimage.util import random_noise


def flip(image):
    return np.fliplr(image)


def bright(image):
    res = image + 0.15
    res[res > 1] = 1
    return res


def noise(image):
    return random_noise(image, var=0.01)


def shear(image):
    flag = bool(random.getrandbits(1))
    r = -0.2 if flag else 0.2
    res = warp(image, AffineTransform(shear=r), order=1, preserve_range=True, mode='wrap')
    return res


def dark(image):
    res = image - 0.15
    res[res < 0] = 0
    return res


def rotation(image):
    flag = bool(random.getrandbits(1))
    r = random.randint(-70, -50) if flag else random.randint(50, 70)
    return rotate(image, angle=r)


def noise_bright(image):
    return noise(bright(image))


def noise_bright_shear(image):
    return noise_bright(shear(image))


def flip_shear_noise(image):
    return flip(shear(noise(image)))


def flip_dark(image):
    return flip(dark(image))


def shear_noise_dark(image):
    return shear(noise(dark(image)))


def flip_shear(image):
    return flip(shear(image))


def flip_noise(image):
    return flip(noise(image))


def flip_rotation(image):
    return flip(rotation(image))


def noise_rotation(image):
    return noise(rotation(image))


def shear_rotation_dark(image):
    return shear(rotation(dark(image)))


def flip_shear_rotation(image):
    return flip(shear(rotation(image)))


def rotation_bright(image):
    return rotation(bright(image))


def rotation_bright_shear(image):
    return rotation(bright(shear(image)))


def flip_rotation_bright_shear(image):
    return rotation_bright_shear(flip(image))


def flip_dark_noise(image):
    return flip_dark(noise(image))


def shear_noise(image):
    return shear(noise(image))



def rotation_shear(image):
    return rotation(shear(image))


STRATEGY_PICKER = {
    'flip': flip,
    'rotation_bright': rotation_bright,
    'bright': bright,
    'shear': shear,
    'dark': dark,
    'rotation_bright_shear': rotation_bright_shear,
    'flip_shear_rotation': flip_shear_rotation,
    'flip_dark': flip_dark,
    'flip_shear_noise': flip_shear_noise,
    'noise': noise,
    'shear_rotation_dark': shear_rotation_dark,
    'flip_shear': flip_shear,
    'flip_rotation': flip_rotation,
    'rotation': rotation,
    'noise_bright_shear': noise_bright_shear,
    'noise_rotation': noise_rotation,
    'flip_rotation_bright_shear': flip_rotation_bright_shear,
    'flip_dark_noise': flip_dark_noise,
    'shear_noise': shear_noise,
    'rotation_shear': rotation_shear
}


def save_augmentation(image, image_name, label, features, data, dst_dir, aug_num=12):
    strategies = list(STRATEGY_PICKER.keys())
    random.shuffle(strategies)
    strategy_arr = strategies[:aug_num]
    for strategy in strategy_arr:
        res = STRATEGY_PICKER[strategy](image)
        save_image(res, 'Aug' + strategy + '_' + image_name, dst_dir)
        data.append(( 'Aug' + strategy + '_' + image_name, features, label))


def save_image(image, image_name, dst_dir):
    result = Image.fromarray((image * 255).astype(np.uint8))
    result.save(os.path.join(dst_dir, image_name))