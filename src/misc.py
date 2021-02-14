import mimetypes
import os


import numpy as np


from src.data import tw_face


IMAGE_EXTENSIONS = [
    ext for ext in mimetypes.types_map if mimetypes.types_map[ext].split('/')[0] == 'image'
]


def is_image(path):
    for ext in IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def list_image(folder):
    images = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not is_image(path):
            continue
        images.append(path)
    return images


def display_result(name_arr, res_arr):
    for name, res in zip(name_arr, res_arr):
        tops = np.argsort(res)

        print(name)
        for i in range(tops.size):
            print('  top-{}: {}, {:.2f}'.format(i+1, tw_face.LABELS[tops[i]], res[tops[i]]))


def softmax(x):
    exp_x = np.exp(x - np.expand_dims(np.amax(x, axis=-1), -1))
    return exp_x / np.expand_dims(exp_x.sum(axis=-1), -1)
