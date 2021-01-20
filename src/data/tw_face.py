"""
Dr. Goh's Dataset for old and young Taiwanese people

young: 334
2-30. 72-91
old: 293

"""

import os
import re

import numpy as np


# format: 0021a00.JPG
# a-f: labels 0-6
FILE_REGEX = r'ck([0-9]{2})[0,2]1([a-g])[0-9]{2}\.png'
DATA_FOLDER = './data'


"""
We need to align the label to FER2013:
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
and original:
    (a: Neutral; b: Happy; c: Sad; d: Angry; e: Disgusted; f: Fearful; g: Surprised).
"""

label_mapping = {
    'a': 6,
    'b': 3,
    'c': 4,
    'd': 0,
    'e': 1,
    'f': 2,
    'g': 5
}


def is_old(subject_idx):
    subject_idx = int(subject_idx)
    return 30 < subject_idx < 72 or subject_idx == 1


def read_data(subset=None):
    old_x = []
    old_y = []
    young_x = []
    young_y = []
    for x in os.listdir(DATA_FOLDER):
        match_obj = re.match(FILE_REGEX, x)
        if match_obj is None:
            continue
        subject_idx = match_obj[1]
        expression = match_obj[2]
        y = label_mapping[expression]
        x = os.path.join(DATA_FOLDER, x)
        if is_old(subject_idx):
            old_x.append(x)
            old_y.append(y)
        else:
            young_x.append(x)
            young_y.append(y)

    old_y = np.array(old_y)
    young_y = np.array(young_y)

    if subset == 'young':
        return young_x, young_y
    if subset == 'old':
        return old_x, old_y

    return old_x + young_x, np.append(old_y, young_y)
