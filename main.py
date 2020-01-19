import argparse
import copy
import os
import pdb
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
import tqdm


SIFT_FEATURES = 2000

ANSWERS_CROP = (0.425, 0.875)
ANSWERS_START_LEFT = (854, 983)
ANSWERS_START_RIGHT = (866, 3168)
ANSWERS_INCREASE = (114.5, 140)
ANSWERS_CELL_SIZE = (60, 80)
ANSWERS_CHOICES = ("A", "B", "C", "D")

NUM_ANSWERS = 15
NUM_VARIANTS = 2
NUM_CHOICES = 4


class Subject:
    UNKNOWN = 0
    INFORMATICA = 1
    FIZICA = 2


class Answers:
    def __init__(self):
        self._answers = [None for _ in range(NUM_VARIANTS * NUM_ANSWERS)]
        self.subject = Subject.UNKNOWN
        self.subject_no = None
    
    def __getitem__(self, idx):
        return self._answers[idx]
    
    def __setitem__(self, idx, value):
        self._answers[idx] = value
    
    def __repr__(self):
        return "\n".join(["{}: {}".format(idx + 1, answer) for idx, answer in enumerate(self._answers)])


def load_image_cv2(path, greyscale=False, noiseless=False):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if greyscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if noiseless:
        image = remove_noise(image, size=5)
    return image


def random_string(length=32):
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def crop(image, crop):
    return image[crop[0]:crop[2], crop[1]:crop[3]]


def remove_noise(image, size=3):
    return scipy.ndimage.median_filter(image, size=size)


def normalize(image, template):
    detector = cv2.xfeatures2d.SIFT_create(SIFT_FEATURES)
    img_kp, img_des = detector.detectAndCompute(image, None)
    tmp_kp, tmp_des = detector.detectAndCompute(template, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(img_des, tmp_des, None)
    matches = sorted(matches, key=lambda match: match.distance)

    img_m = np.zeros((len(matches), 2), dtype=np.float32)
    tmp_m = np.zeros((len(matches), 2), dtype=np.float32)
    for idx, match in enumerate(matches):
        img_m[idx] = img_kp[match.queryIdx].pt
        tmp_m[idx] = tmp_kp[match.trainIdx].pt
    homography, _ = cv2.findHomography(img_m, tmp_m, cv2.RANSAC)
    img = cv2.warpPerspective(image, homography, template.shape[::-1])

    return img


def visualize_answers(image, answers):
    iy, ix = ANSWERS_INCREASE
    dy, dx = ANSWERS_CELL_SIZE
    image_o = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for k, (sy, sx) in enumerate([ANSWERS_START_LEFT, ANSWERS_START_RIGHT]):
        for i in range(15):
            pts = [(int(sy + iy * i - dy/2), int(sx + ix * j - dx/2)) for j in range(NUM_CHOICES)]
            best = ANSWERS_CHOICES.index(answers[k * NUM_ANSWERS + i])
            for j, (y, x) in enumerate(pts):
                cv2.rectangle(image_o, (x, y), (x + dx, y + dy), (0, 255, 0,) if j == best else (255, 0, 0), 1)
    plt.imshow(image_o)
    plt.show()


def crop_answers(image):
    height, width = image.shape
    cy, cx = ANSWERS_CROP
    image = crop(image, (int(cy * height), 0, int(cx * height), width))
    return image


def get_answers(image):
    answers = Answers()
    iy, ix = ANSWERS_INCREASE
    dy, dx = ANSWERS_CELL_SIZE
    for k, (sy, sx) in enumerate([ANSWERS_START_LEFT, ANSWERS_START_RIGHT]):
        for i in range(NUM_ANSWERS):
            pts = [(int(sy + iy * i - dy/2), int(sx + ix * j - dx/2)) for j in range(NUM_CHOICES)]
            sums = [np.sum(image[y:y+dy, x:x+dx]) for (y, x) in pts]
            best = np.argmin(sums)
            answers[k * NUM_ANSWERS + i] = ANSWERS_CHOICES[best]

    return answers


def main(images, template):
    template = load_image_cv2(template, True, True)
    images = sorted(images)
    for idx, path in enumerate(tqdm.tqdm(images)):
        image = load_image_cv2(path, True, True)
        image = normalize(image, template)
        image = crop_answers(image)
        answers = get_answers(image)
        visualize_answers(image, answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, nargs="+", help="Paths to input images to grade.")
    parser.add_argument("--template", type=str, help="Path to image to use as template for matching.")
    args, _ = parser.parse_known_args()
    main(**vars(args))