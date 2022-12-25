import os
import re
from skimage.feature import hog, local_binary_pattern, haar_like_feature
from sklearn.model_selection import cross_val_score
import numpy as np


def read_image(filename):
    with open(filename, "rb") as f:
        buffer = f.read()
    header, width, height, max_val = re.search(b"(^P5\\s(\\d+)\\s(\\d+)\\s(\\d+)\\s)", buffer).groups()

    return np.frombuffer(buffer, dtype='u1' if int(max_val) < 256 else '>u2',
                         count=int(width) * int(height), offset=len(header)
                         ).reshape((int(height), int(width)))


def get_all_images():
    images = os.listdir("dataset/")
    all_images = []
    all_images += [i for i in images if i[-4:] == '.pgm']
    filter_images = []
    for image in all_images:
        filter_images.append([image, read_image("dataset/{}".format(image))])

    return filter_images


def preprocessing(images, label_flag="name", feature_flag="hog"):
    if label_flag == "name":
        vals = ['megak', 'night', 'glickman', 'cheyer', 'an2i', 'bpm',
                'saavik', 'kk49', 'tammo', 'steffi', 'boland', 'mitchell',
                'sz24', 'danieln', 'karyadi', 'ch4f', 'kawamura', 'phoebe',
                'at33', 'choon']
        idx = 0
    elif label_flag == "position":
        vals = ['left', 'right', 'up', 'straight']
        idx = 1
    elif label_flag == "expression":
        vals = ['neutral', 'happy', 'sad', 'angry']
        idx = 2
    elif label_flag == "glasses":
        vals = ['open', 'sunglasses']
        idx = 3
    else:
        raise ValueError("There is no %s flag for encoding label" % label_flag)

    labels = [vals.index(i[0].split("_")[idx]) for i in images]

    if feature_flag == "hog":
        features = [hog(i[1]) for i in images]
    elif feature_flag == "lbp":
        features = [local_binary_pattern(i[1], 7, 1.0) for i in images]
    elif feature_flag == "haar":
        features = [haar_like_feature(i[1], 0, 0, 20, 20) for i in images]
    else:
        raise ValueError("There is no %s flag for feature processing" % feature_flag)
    return labels, features


def cross_validation(train, test, model):
    scores = cross_val_score(model, train, test, cv=10)
    return scores.mean(), scores.std()

