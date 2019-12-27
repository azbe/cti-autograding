import argparse
import copy
import pdb
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm


def load_image(path):
    image = Image.open(path)
    image = image.convert("L")
    image = np.array(image)
    image = image / 255.0
    return image


def save_image(image, path):
    image = image * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)


def get_random_string(length=32):
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def crop(image, crop):
    return image[crop[0]:crop[2], crop[1]:crop[3]]


def binarize(image, threshold=127):
    image = copy.deepcopy(image)
    image[image <= (threshold / 255.0)] = 0.0
    image[image > (threshold / 255.0)] = 1.0
    return image


def gradient(image, direction):
    rows, cols = image.shape
    if direction == "right":
        grad = image[:, 1:] - image[:, :cols-1]
    elif direction == "left":
        grad = image[:, :cols-1] - image[:, 1:]
    elif direction == "down":
        grad = image[1:, :] - image[:rows-1, :]
    elif direction == "top":
        grad = image[:rows-1, :] - image[1:, :]
    else:
        raise ValueError
    grad = (grad + 1.0) / 2.0
    return grad


def get_answers(image):
    ymin, xmin, ymax, xmax = int(0.425 * image.shape[0]), 0, int(0.9 * image.shape[0]), image.shape[1]
    image = crop(image, (ymin, xmin, ymax, xmax))
    image = binarize(image, threshold=175)


def main(images):
    for image in tqdm.tqdm(images):
        image = load_image(image)
        get_answers(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, nargs="+", help="Paths to input images to grade.")
    args, _ = parser.parse_known_args()
    main(**vars(args))